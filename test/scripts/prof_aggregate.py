#!/usr/bin/env python3

import sys
import os
import argparse
import json
import pandas as pd

def parse_trace_entry(entry, datatype, count):
    """Parse a single trace entry and return operation details if it matches criteria."""
    if not ((entry['cat'] == 'COLL' or entry['cat'] == 'P2P') and entry['ph'] == 'b'):
        return None

    if entry['args']['Datatype'] != datatype or entry['args']['Count'] != int(count):
        return None

    return {
        'rank': entry['args']['Rank'],
        'func': entry['name'],
        'datatype': entry['args']['Datatype'],
        'count': entry['args']['Count'],
        'algo': entry['args']['Algorithm'],
        'proto': entry['args']['Protocol']
    }

def parse_proxy_times(entry, proxy_times):
    """Parse timing information from proxy entries."""
    if entry['ph'] == 'b':
        if entry['name'] == 'SendGpuWait' or entry['name'] == 'RecvWait':
            proxy_times[0] -= float(entry['ts'])
        elif entry['name'] == 'SendPeerWait' or entry['name'] == 'RecvFlushWait':
            proxy_times[1] -= float(entry['ts'])
        elif entry['name'] == 'SendWait' or entry['name'] == 'RecvGpuWait':
            proxy_times[2] -= float(entry['ts'])
    elif entry['ph'] == 'e':
        if entry['name'] == 'SendGpuWait' or entry['name'] == 'RecvWait':
            proxy_times[0] += float(entry['ts'])
        elif entry['name'] == 'SendPeerWait' or entry['name'] == 'RecvFlushWait':
            proxy_times[1] += float(entry['ts'])
        elif entry['name'] == 'SendWait' or entry['name'] == 'RecvGpuWait':
            proxy_times[2] += float(entry['ts'])

def parse_kernel_times(entry, kernel_time):
    """Parse timing information from the GPU entries."""
    if entry['ph'] == 'b':
        kernel_time[0] = float(entry['args']['StopGpuClk'] - entry['args']['StartGpuClk'])*1e-3

def create_dataframe_row(op_details, channel, peer, proxy_times, kernel_time, is_send=True):
    """Create a DataFrame row with the given operation details and timing information."""
    base_row = {
        "rank": op_details['rank'],
        "func": op_details['func'],
        "datatype": op_details['datatype'],
        "count": op_details['count'],
        "algo": op_details['algo'],
        "proto": op_details['proto'],
        "channel": channel,
        "peer": peer,
        "Kernel": kernel_time[0],
        "ProxyOp": proxy_times[3]
    }

    if is_send:
        base_row.update({
            "SendGpuWait": proxy_times[0],
            "SendPeerWait": proxy_times[1],
            "SendWait": proxy_times[2],
            "RecvWait": 0.0,
            "RecvFlushWait": 0.0,
            "RecvGpuWait": 0.0,
        })
    else:
        base_row.update({
            "SendGpuWait": 0.0,
            "SendPeerWait": 0.0,
            "SendWait": 0.0,
            "RecvWait": proxy_times[0],
            "RecvFlushWait": proxy_times[1],
            "RecvGpuWait": proxy_times[2],
        })

    return base_row

def build_dataframe(filename, datatype, count, opnum):
    """Build a pandas DataFrame from trace files with timing information."""
    columns = [
        "rank", "func", "datatype", "count", "algo", "proto", "channel", "peer", "Kernel", "ProxyOp",
        "SendGpuWait", "SendPeerWait", "SendWait", "RecvWait", "RecvFlushWait", "RecvGpuWait"
    ]
    df = pd.DataFrame(columns=columns)

    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    files = [f for f in os.listdir(dirname) if f.startswith(basename)]

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)

        op_details = None
        channel = peer = 0
        proxy_times = [0.0, 0.0, 0.0, 0.0]
        kernel_time = [0.0]
        opcount = 0

        for entry in data:
            # account for empty record {}
            if 'cat' not in entry: continue

            tmp = parse_trace_entry(entry, datatype, count)
            if tmp is not None:
                if opnum == -1 or opcount == opnum:
                    op_details = tmp
                else:
                    tmp = {}
                opcount += 1
                continue

            # parse GPU entry record
            if op_details and entry['cat'] == 'GPU' and entry['ph'] == 'b':
                parse_kernel_times(entry, kernel_time)
                channel = entry['args']['Channel']
                continue

            elif op_details and entry['cat'] == 'GPU' and entry['ph'] == 'e':
                if kernel_time[0] > 0.0:
                    row = create_dataframe_row(
                        op_details, channel, 0, [0.0, 0.0, 0.0, 0.0], kernel_time,
                        is_send=0
                    )
                    df.loc[len(df)] = row
                    kernel_time = [0.0]

            # parse PROXY entry record
            elif op_details and entry['cat'] == 'PROXY' and (entry['name'] == 'ProgressSend' or entry['name'] == 'ProgressRecv') and entry['ph'] == 'b':
                channel = entry['args']['Channel']
                peer = entry['args']['Peer']
                proxy_times[3] -= float(entry['ts'])

            elif op_details and entry['cat'] == 'PROXY' and (entry['name'] == 'ProgressSend' or entry['name'] == 'ProgressRecv') and entry['ph'] == 'e':
                proxy_times[3] += float(entry['ts'])
                row = create_dataframe_row(
                    op_details, channel, peer, proxy_times, kernel_time,
                    is_send=(entry['name'] == 'ProgressSend')
                )
                df.loc[len(df)] = row
                proxy_times = [0.0, 0.0, 0.0, 0.0]
                kernel_time = [0.0]

            # parse NET entry record
            elif op_details and entry['cat'] == 'NET':
                parse_proxy_times(entry, proxy_times)

            elif op_details and (entry['cat'] == 'COLL' or entry['cat'] == 'P2P') and entry['ph'] == 'e':
                op_details = None

    return df

def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="The basename of the trace files generated by the profiler")
    parser.add_argument("-d", "--datatype", help="Operation datatype")
    parser.add_argument("-c", "--count", help="Operation count")
    parser.add_argument("-n", "--opnum", help="Operation number from top of the log", default=-1)
    parser.add_argument("-o", "--output", help="Output filename for aggregated data")
    parser.add_argument("-p", "--print", help="Print data to screen", action='store_true')

    args = parser.parse_args()
    df = build_dataframe(args.filename, args.datatype, int(args.count), int(args.opnum))

    if args.output:
        df.to_csv(f"{args.output}.csv", index=False)

    if args.print:
        print(df)

if __name__ == "__main__":
    main()
