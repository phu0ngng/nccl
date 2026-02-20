#!/usr/bin/env python3
"""
Kibana Integration Wrapper for NCCL Inspector Performance Summary Exporter

This wrapper provides internal integrations (pyslurm, nvdataflow) around the core
exporter functionality. It imports the external exporter and extends it with
internal system capabilities.
"""

import sys
import os
from pathlib import Path
import logging
import pyslurm
from nvdataflow import post
from datetime import datetime
import configparser
import json
import math

# Add the external exporter to the path
external_exporter_path = Path(__file__).parent.parent.parent / "inspector" / "exporter" / "example"
sys.path.insert(0, str(external_exporter_path))

# Import the external exporter functions
try:
    from perf_summary_exporter import (
        create_per_node_parquet_files,
        bytes_to_human_readable,
        summarize_data_per_comm_coll_type,
        generate_visualizations
    )
except ImportError as e:
    print(f"Error importing external exporter: {e}")
    print(f"Make sure the external exporter is available at: {external_exporter_path}")
    sys.exit(1)

def calculate_batch_size(sample_record, max_size_bytes=18000000):
    """Calculate optimal batch size to stay under the size limit"""
    # Use 18MB limit to leave buffer for JSON overhead
    sample_json = json.dumps(sample_record)
    record_size = len(sample_json.encode('utf-8'))

    if record_size == 0:
        return 1000  # Default batch size if calculation fails

    # Calculate how many records can fit in the size limit
    max_records = max_size_bytes // record_size

    # Ensure minimum batch size of 1 and maximum of 10000 for performance
    batch_size = max(1, min(max_records, 10000))

    logging.info(f"Calculated batch size: {batch_size} records (estimated {record_size} bytes per record)")
    return batch_size


def upload_data_in_batches(bulk_data, project, description="data"):
    """Upload data in batches to stay within size limits"""
    if not bulk_data:
        logging.warning(f"No {description} to upload")
        return

    # Calculate optimal batch size
    batch_size = calculate_batch_size(bulk_data[0])
    total_records = len(bulk_data)
    num_batches = math.ceil(total_records / batch_size)

    logging.info(f"Uploading {total_records} {description} records in {num_batches} batches (batch size: {batch_size})")

    successful_batches = 0
    failed_batches = 0

    # Import tqdm for progress bar
    from tqdm.auto import tqdm

    # Create progress bar
    progress_desc = f"Uploading {description}"
    with tqdm(total=num_batches, desc=progress_desc, unit="batch") as pbar:
        for i in range(0, total_records, batch_size):
            batch = bulk_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            try:
                logging.info(f"Uploading batch {batch_num}/{num_batches} with {len(batch)} records")
                post(data=batch, project=project)
                successful_batches += 1
                logging.info(f"Batch {batch_num}/{num_batches} uploaded successfully")
                pbar.set_postfix({
                    'Success': successful_batches,
                    'Failed': failed_batches,
                    'Records': f"{batch_num * batch_size}/{total_records}"
                })
            except Exception as e:
                failed_batches += 1
                logging.error(f"Failed to upload batch {batch_num}/{num_batches}: {e}")
                pbar.set_postfix({
                    'Success': successful_batches,
                    'Failed': failed_batches,
                    'Records': f"{batch_num * batch_size}/{total_records}"
                })

            # Update progress bar
            pbar.update(1)

    logging.info(f"Upload complete: {successful_batches} successful, {failed_batches} failed out of {num_batches} batches")

    if failed_batches > 0:
        raise Exception(f"Failed to upload {failed_batches} out of {num_batches} batches")


def setup_logging(output_dir):
    """Setup logging to file only"""
    log_file = output_dir / "output.log"

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler only
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)

    logging.info(f"Logging configured - writing to {log_file}")
    print(f"Logging configured - writing to {log_file}")  # Explicit stdout notification

def get_job_details(jobid):
    jobids = [int(jobid)]
    job_filter = pyslurm.db.JobFilter(ids=jobids)
    jobs = pyslurm.db.Jobs.load(job_filter)
    job_data = {}
    for job_id, job_info in jobs.items():
        job_dict = {}
        available_attrs = dir(job_info)
        for attr in [
                "name",
                "cluster",
                "state",
                "partition",
                "user_name",
                "start_time",
                "end_time",
                "submit_time",
                "time_limit",
                "num_tasks",
                "num_cpus",
                "memory_requested",
                "nodes",
                "node_list",
        ]:
            if attr in available_attrs:
                job_dict[attr] = getattr(job_info, attr, "N/A")
                job_data[job_id] = job_dict
    return job_data

def upload_to_kibana(df, comm_type, coll_type, jobid, config):
    """Upload detailed data to Kibana using nvdataflow (internal functionality)"""
    logging.info(f"Uploading data to Kibana for jobid {jobid}: {comm_type} and {coll_type}")

    job_data = get_job_details(jobid)
    job_data = job_data[int(jobid)]

    if not job_data:
        logging.error(f"No job data found for job ID: {jobid}")
        return
    cluster_name = job_data['cluster']
    user_name = job_data['user_name']

    bulk_data = []
    for _, row in df.iterrows():
        n_ranks = row['n_ranks'][0]
        msg_size_human_readable = row['human_readable_coll_msg_size_bytes']
        tag = f"{msg_size_human_readable}_r_{n_ranks}"
        jobid_tag = f"{jobid}-{tag}"
        coll_sn = row['coll_sn']
        msg_size = row['coll_msg_size_bytes']

        data = {
            "_id": f"{cluster_name}-{jobid}-{tag}-{comm_type}-{coll_type}-{coll_sn}-{msg_size}",
            "s_jobid": f"{jobid}",
            "s_cluster": f"{cluster_name}",
            "s_user": user_name,
            "s_tag": tag,
            "s_comm_type": comm_type,
            "s_coll_type": coll_type,
            "ts_current_time": int(datetime.now().timestamp() * 1000),
            "d_mean_coll_busbw": row['mean_coll_busbw_gbs'],
            "d_coll_sn": coll_sn,
            "d_msg_size": msg_size,
            "s_msg_size_human_readable": msg_size_human_readable,
            "s_jobid_tag": jobid_tag,
            "d_coll_start_timestamp_us": row['coll_start_timestamp_us'],
            "d_coll_end_timestamp_us": row['coll_end_timestamp_us'],
            "d_coll_duration_us": row['coll_duration_us'],
        }
        bulk_data.append(data)

    # Post all data to Kibana in batches
    try:
        kibana_project = config.get('kibana', 'kibana_project_detailed')
        upload_data_in_batches(bulk_data, kibana_project, f"detailed data for job {jobid}: {comm_type} and {coll_type}")
        logging.info(f"All detailed data for job {jobid} uploaded successfully to Kibana")
    except Exception as e:
        logging.error(f"Failed to upload detailed data to Kibana: {e}")


def upload_summary_to_kibana(df, comm_type, coll_type, jobid, config):
    """Upload summary data to Kibana using nvdataflow (internal functionality)"""
    logging.info(f"Uploading summary data to Kibana for jobid {jobid}: {comm_type} and {coll_type}")

    job_data = get_job_details(jobid)
    job_data = job_data[int(jobid)]

    if not job_data:
        logging.error(f"No job data found for job ID: {jobid}")
        return
    cluster_name = job_data['cluster']
    user_name = job_data['user_name']

    # Group by both coll_msg_size_bytes and comm_type and compute summary statistics
    summary_df = df.groupby("coll_msg_size_bytes").agg(
        avg_busbw=("mean_coll_busbw_gbs", "mean"),
        max_busbw=("mean_coll_busbw_gbs", "max"),
        min_busbw=("mean_coll_busbw_gbs", "min"),
        op_count=("coll_sn", "count"),
        n_ranks=("n_ranks", "first")
    ).reset_index()

    bulk_data = []
    for _, row in summary_df.iterrows():
        n_ranks = row['n_ranks'][0] if row['n_ranks'] else 'unknown'
        msg_size = row['coll_msg_size_bytes']
        msg_size_human_readable = bytes_to_human_readable(msg_size)
        tag = f"{msg_size_human_readable}_r_{n_ranks}"

        data = {
            "_id": f"{cluster_name}-{jobid}-{tag}-{comm_type}-{coll_type}-{msg_size}",
            "s_jobid": f"{jobid}",
            "s_cluster": f"{cluster_name}",
            "s_user": user_name,
            "s_tag": tag,
            "s_comm_type": comm_type,
            "s_coll_type": coll_type,
            "s_msg_size_human_readable": msg_size_human_readable,
            "d_msg_size": msg_size,
            "d_avg_busbw": row['avg_busbw'],
            "d_max_busbw": row['max_busbw'],
            "d_min_busbw": row['min_busbw'],
            "d_op_count": row['op_count'],
            "ts_current_time": int(datetime.now().timestamp() * 1000),
        }
        bulk_data.append(data)
        logging.info(f"Appended record for comm_type: {comm_type}, coll_type: {coll_type}, "
                     f"msg_size_human_readable: {msg_size_human_readable}. "
                     f"Total records appended: {len(bulk_data)}")

    # Post all summary data to Kibana in batches
    try:
        kibana_project_summary = config.get('kibana', 'kibana_project_summary')
        upload_data_in_batches(bulk_data, kibana_project_summary, f"summary data for job {jobid}: {comm_type} and {coll_type}")
        logging.info(f"Summary data for job {jobid} uploaded successfully to Kibana")
    except Exception as e:
        logging.error(f"Failed to upload summary data to Kibana: {e}")


def enhanced_generate_summary(output_root, comm_type, coll_type, upload, jobid, config):
    """Enhanced summary generation with Kibana upload capability"""
    logging.info(f"Generating enhanced summary for jobid: {jobid}, {comm_type} and {coll_type}")

    # Step 1: Summarize data per communication and collective type using the refactored function
    df = summarize_data_per_comm_coll_type(output_root, comm_type, coll_type, f"insp-summary-jobid-{jobid}")

    # Step 2: Upload to Kibana if data exists and upload is requested
    if df is not None and upload:
        upload_to_kibana(df, comm_type, coll_type, jobid, config)
        upload_summary_to_kibana(df, comm_type, coll_type, jobid, config)

    # Step 3: Generate visualizations if data exists
    if df is not None:
        generate_visualizations(df, output_root, comm_type, coll_type)


def enhanced_generate_summary_worker(args):
    """Worker wrapper that sets up logging for multiprocessing"""
    output_root, comm_type, coll_type, upload, jobid, config = args

    # Setup logging for this worker process (file only)
    log_file = output_root / "output.log"
    worker_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [Worker] - %(message)s")

    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # File handler only (append mode for workers)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(worker_formatter)
    file_handler.setLevel(logging.INFO)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)

    # Call the actual function
    return enhanced_generate_summary(output_root, comm_type, coll_type, upload, jobid, config)


def setup_config_and_args():
    """Setup configuration and command line arguments"""
    import argparse
    from pathlib import Path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NCCL Inspector Performance Summary Exporter with Kibana Integration")
    parser.add_argument(
        "--jobid",
        type=str,
        required=True,
        help="The SLURM job ID to process"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to Kibana"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="kibana_config.ini",
        help="Path to configuration file (default: kibana_config.ini)"
    )

    args = parser.parse_args()

    # Load configuration from file
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent / args.config

    if config_path.exists():
        config.read(config_path)
        print(f"Loaded configuration from {config_path}")  # Use print until logging is configured
    else:
        print(f"Configuration file {config_path} not found. Using defaults.")  # Use print until logging is configured
        # Fallback to defaults if config file doesn't exist
        config.add_section('paths')
        config.set('paths', 'default_log_root', '/home/svc-nccl-inspector/nccl-inspector/logs/user-jobs')

        config.add_section('kibana')
        config.set('kibana', 'kibana_project_detailed', 'aidot-fact-nccl-inspector')
        config.set('kibana', 'kibana_project_summary', 'aidot-fact-nccl-inspector-summary')

    return args, config


def setup_directories_and_find_logs(jobid, config):
    """Setup output directory and find log files for the given job ID"""
    from pathlib import Path
    import glob

    # Setup output directory
    output_dir = Path(f"{jobid}-insp")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    # Construct log file path using job ID
    log_root = config.get('paths', 'default_log_root')
    job_insp_dir = Path(log_root) / jobid

    if not job_insp_dir.exists():
        logging.error(f"Job inspection directory not found: {job_insp_dir}")
        sys.exit(1)

    # Find all log files
    logfiles = list(glob.iglob(str(Path(job_insp_dir) / "**" / "*.log"), recursive=True))
    gzlogfiles = list(
        glob.iglob(str(Path(job_insp_dir) / "**" / "*.log.gz"), recursive=True)
    )
    jsonlfiles = list(
        glob.iglob(str(Path(job_insp_dir) / "**" / "*.jsonl"), recursive=True)
    )
    gzjsonlfiles = list(
        glob.iglob(str(Path(job_insp_dir) / "**" / "*.jsonl.gz"), recursive=True)
    )
    if (
            sum((1 for x in [logfiles, gzlogfiles, jsonlfiles, gzjsonlfiles] if len(x) > 0))
            > 1
    ):
        ### TODO: we could probably generate some logic to pick the "right" file to load, but for now, bail
        logging.critical("Appear to have mixed .log/.log.gz/.jsonl/.jsonl.gz; bailing!")
        sys.exit(1)

    files = logfiles + gzlogfiles + jsonlfiles + gzjsonlfiles

    if not files:
        print("No inspector logs found")
        sys.exit(1)

    print(f"Number of log files found: {len(files)}")
    print(f"Job ID: {jobid}")
    print(f"Output directory: {output_dir}")

    return output_dir, files


def main():
    """Main function for the Kibana wrapper"""
    print("Kibana wrapper started")

    # Setup configuration and arguments
    args, config = setup_config_and_args()

    # Setup directories and find log files (this will enhance logging with file output)
    output_dir, files = setup_directories_and_find_logs(args.jobid, config)

    # Create per-node parquet files
    create_per_node_parquet_files(files, output_dir)

    # Generate summaries for all communication and collective types
    comm_types = ["single-rank", "nvlink-only", "hca-only", "mixed"]
    coll_types = ["AllReduce", "AllGather", "ReduceScatter", "Broadcast"]

    summary_args = [
        (output_dir, comm_type, coll_type, args.upload, args.jobid, config)
        for comm_type in comm_types
        for coll_type in coll_types
    ]

    max_workers = min(64, len(summary_args), os.cpu_count() or 1)
    from concurrent.futures import ProcessPoolExecutor
    from tqdm.auto import tqdm

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(enhanced_generate_summary_worker, summary_args),
                total=len(summary_args),
                desc="Generating enhanced summaries",
            )
        )

    print("Enhanced analysis with Kibana integration completed!")


if __name__ == "__main__":
    main()
