#!/usr/bin/env python3
# This file exists to take HugeCTR (optional) followed by one or more NCCL perf test outputs from a single text file and creates a CSV for easy Excel consumption
# HugeCTR cases are detected via a line matching: "<intra/inter/intra+inter> A2A bench:" (default test bench output)
# NCCL test cases are detected via a line matching "CASE: <Case Name>" before each NCCL run in the input file
# This expects HugeCTR to be run first, followed by NCCL
# This currently maps by size on each run only, and expects all cases to be run on the same sizes.
# To additionally map by datatype and redop, uncomment the lines in parse_file()
#
# Example scenario;
# cd ~/nccl-container/
# sbatch scripts/huge_ctr_benchmark.sub
# ../nccl/test/scripts/make_csv.py file=nccl-a2a:2x8_12345678.out
# On Windows
# scp selene:~/nccl-container/nccl-a2a_2x8_12345678.out.csv .
# Open in Excel
import sys
import csv

def help():
  print("Usage: make_csv.py file=<Perf Data Filename> capture=<latency/bw, default=latency> verbose=<True/False, default=False>")
  exit(1)

if '-h' in sys.argv or '--help' in sys.argv[1:]:
  help()

kwnames = ['file','verbose','capture']
kwargs = {x[:x.index('=')]:x[x.index('=')+1:] for x in sys.argv[1:] if '=' in x and x[:x.index('=')] in kwnames}

if 'file' not in kwargs: help()
file = kwargs['file']
csv_file = file.replace(":","_") + ".csv" # Replace : with _ for Windows filename compatability (: requried for selene job name)
verbose = bool(kwargs.get('verbose', False))
capture = kwargs.get('capture', 'latency')
inp_field_index=9 # Time (latency)
oop_field_index=5 # Time (latency)
if capture == 'bw':
    inp_field_index=11 # Bus BW
    oop_field_index=7  # Bus BW

exit_code = 0
keys = {}

def parse_hugectr(cases):
  f = open(file, "r")
  out = f.read()
  test_case = ""
  for ln in (out or '').split('\n'):
    ln = ln.rstrip()
    if 'A2A bench:' in ln:
      test_case = "HugeCTR " + ln.split("A2A bench:",1)[0]
      cases[test_case] = {}
      cases[test_case]["inp_times"] = {}
      cases[test_case]["oop_times"] = {}
    elif ln.startswith("size(in B)  time(in us)"):
      continue
    elif ln.startswith('CASE: '):
      # Start of NCCL test cases
      return
    elif test_case != "" and ln != '':
      try:
        size = int(ln.split(" ",1)[0])
        time = float(ln.split(" ",1)[1])
        if verbose:
          print(test_case + " " + str(size) + " " + str(time))
        cases[test_case]["oop_times"][size] = time
        cases[test_case]["inp_times"][size] = time
      except:
        pass

def parse_nccl(cases):
  test_case = ""
  f = open(file, "r")
  out = f.read()
  for ln in (out or '').split('\n'):
    ln = ln.rstrip()
    if ln.startswith('#'):
      # This line precedes the begin of the performance data
      if ln.split()[:4] == ['#','size','count','type']:
        cuts=[]
        for i in range(1,len(ln)):
          if ln[i-1] not in (' ','#') and ln[i]==' ':
            cuts += [i]
        cuts = [0] + cuts + [len(ln)]
        def field(ln,i):
          return ln[cuts[i]:cuts[i+1]].strip()
    elif ln.startswith('CASE: '):
      test_case = ln.split("CASE: ",1)[1]
      cases[test_case] = {}
      cases[test_case]["inp_times"] = {}
      cases[test_case]["oop_times"] = {}
    elif ln != '':
      try:
        size = int(field(ln,0))
        # dtype = field(ln,2)
        # redop = field(ln,3)
        key = size # (size,dtype,redop)
        keys[key] = 1
        cases[test_case]["oop_times"][key] = (float(field(ln,oop_field_index))) # out-of-place time
        cases[test_case]["inp_times"][key] = (float(field(ln,inp_field_index))) # in-place time
      except:
        pass

cases = {}
parse_hugectr(cases)
parse_nccl(cases)

# Write to CSV
with open(csv_file, 'w', encoding='ASCII', newline='') as f:
    writer = csv.writer(f)

    header_row = ["Size"]
    for case in cases.keys():
      header_row.append(case)

    writer.writerow(header_row)
    for key in keys.keys():
      row = [key]
      for case in cases.keys():
        try:
          row.append(cases[case]["oop_times"][key])
        except:
          print("Key not found: ")
          print("case: " + case)
          print("key: " + str(key))
          row.append(0)

      # write row
      writer.writerow(row)
