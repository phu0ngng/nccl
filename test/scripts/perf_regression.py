#!/usr/bin/env python3
import sys
import statistics

def help():
  print("Usage: perf_regression.py old=<baseline_perf_file> new=<new_perf_file> threshold=<percent, default=10> rmad_threshold=<float, default=0.2> verbose=<True/False, default=False>")
  print("This file will parse the old and new perf outputs and calculates the average, median, mad, rmad and stddev of each")
  print("It will then pass or fail based on the rmad and median of the new set of runs vs. the old.")
  exit(1)

if '-h' in sys.argv or '--help' in sys.argv[1:]:
  help()

kwnames = ['new','old','rmad_threshold','threshold','verbose']
kwargs = {x[:x.index('=')]:x[x.index('=')+1:] for x in sys.argv[1:] if '=' in x and x[:x.index('=')] in kwnames}

if 'new' not in kwargs: help()
file_new = kwargs['new']
if 'old' not in kwargs: help()
file_old = kwargs['old']
threshold = float(kwargs.get('threshold', 10))
verbose = bool(kwargs.get('verbose', False))
rmad_threshold = float(kwargs.get('rmad_threshold', 0.2))
exit_code = 0

def format_float(n):
  return "%.2f"%n

def print_stats(t0_stats, t1_stats):
  print('{:>20}: {:>9} {:>9}'.format("Iteration", "Baseline", "New"))
  for i in range(len(t0_stats["times"])):
    line = '{:>20}: {:>9} {:>9}'.format(i, format_float(t0_stats["times"][i]),format_float(t1_stats["times"][i]))
    print(line)
  line = '{:>20}: {:>9} {:>9}'.format("StdDev", format_float(t0_stats["stddev"]),format_float(t1_stats["stddev"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("MAD", format_float(t0_stats["mad"]),format_float(t1_stats["mad"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("RMAD", format_float(t0_stats["rmad"]),format_float(t1_stats["rmad"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("Median", format_float(t0_stats["median"]),format_float(t1_stats["median"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("Average", format_float(t0_stats["avg"]),format_float(t1_stats["avg"]))
  print(line)

# In statistics, the median absolute deviation (MAD) is a robust measure of the variability of a univariate sample of quantitative data.
# The variance and standard deviation are also measures of spread, but they are more affected by extremely high or extremely low values and non normality.
def mad(stats):
  return statistics.median([abs(time - stats["median"]) for time in stats["times"]])

def compare_times(placeness, bads, wins, neutrals, new_times, old_times, rmad_threshold):
  for (size,dtype,redop),t0 in old_times.items():
    try:
      t1 = new_times[size,dtype,redop]

      # t1 and t0 are arrays of times (in microseconds) that should be of equal length
      if len(t0) != len(t1):
        print("Mismatched number of runs. len(baselines): " + len(t0) + " != len(new): " + len(t1))
        exit_code = 1
        return

      # Store stats for comparison
      t1_stats = {}
      t0_stats = {}
      t1_stats["avg"]    = sum(t1) / len(t1)
      t1_stats["times"]  = t1
      t0_stats["avg"]    = sum(t0) / len(t0)
      t0_stats["times"]  = t0

      # We already enforced same length
      if len(t0) > 1:
        t1_stats["stddev"] = statistics.stdev(t1)
        t1_stats["median"] = statistics.median(t1)
        t1_stats["mad"]    = mad(t1_stats)
        t1_stats["rmad"]   = t1_stats["mad"] / t1_stats["median"]
        t0_stats["stddev"] = statistics.stdev(t0)
        t0_stats["median"] = statistics.median(t0)
        t0_stats["mad"]    = mad(t0_stats)
        t0_stats["rmad"]   = t0_stats["mad"] / t0_stats["median"]
      else:
        t1_stats["stddev"] = 0
        t1_stats["median"] = t1[0]
        t1_stats["mad"]    = 0
        t1_stats["rmad"]   = 0
        t0_stats["stddev"] = 0
        t0_stats["median"] = t0[0]
        t0_stats["mad"]    = 0
        t0_stats["rmad"]   = 0

      gain = 100*(t1_stats["median"] - t0_stats["median"])/t0_stats["median"]
      if t1_stats["rmad"] > rmad_threshold:
        print("WARNING: High variance for " + placeness + " detected. New rmad=" + format_float(t1_stats["rmad"]) + ", rmad_threshold=" + format_float(rmad_threshold))

      if t0_stats["rmad"] > rmad_threshold:
        print("WARNING: High variance for " + placeness + " detected. Baseline rmad=" + format_float(t0_stats["rmad"]) + ", rmad_threshold=" + format_float(rmad_threshold))

      if gain >= threshold:
        bads += [(placeness, gain,size,dtype,redop,t0_stats,t1_stats)]
      elif gain <= -threshold:
        wins += [(placeness, gain,size,dtype,redop,t0_stats,t1_stats)]
      else:
        neutrals += [(placeness, gain,size,dtype,redop,t0_stats,t1_stats)]
    except KeyError:
      bads += [('FAILED',size,dtype,redop)]

def parse_file(file_name, oop_times, inp_times):
  f = open(file_name, "r")
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
    elif ln != '':
      try:
        size = int(field(ln,0))
        dtype = field(ln,2)
        redop = field(ln,3)
        key = (size,dtype,redop)
        if key not in oop_times.keys():
          oop_times[key] = []
        if key not in inp_times.keys():
          inp_times[key] = []

        oop_times[key].append(float(field(ln,5))) # out-of-place time
        inp_times[key].append(float(field(ln,9))) # in-place time
      except:
        pass

new_oop_times = {}
new_inp_times = {}
parse_file(file_new, new_oop_times, new_inp_times)

old_oop_times = {}
old_inp_times = {}
parse_file(file_old, old_oop_times, old_inp_times)

bads=[]
wins=[]
neutrals = []
compare_times("out-of-place", bads, wins, neutrals, new_oop_times, old_oop_times, rmad_threshold)
compare_times("in-place", bads, wins, neutrals, new_inp_times, old_inp_times, rmad_threshold)

bads.sort(key=lambda x:(str(type(x[0])),x[0]), reverse=True)
wins.sort(key=lambda x:(str(type(x[0])),x[0]), reverse=False)

def format_bytes(n):
  if n < 1<<10:
    return "%gB"%n
  elif n < 1<<20:
    return "%.3gK"%(n/(1<<10))
  elif n < 1<<30:
    return "%.3gM"%(n/(1<<20))
  else:
    return "%.3gG"%(n/(1<<30))

def format_float(n):
  return "%.2f"%n

def print_stats(t0_stats, t1_stats):
  print('{:>20}: {:>9} {:>9}'.format("Iteration", "Baseline", "New"))
  for i in range(len(t0_stats["times"])):
    line = '{:>20}: {:>9} {:>9}'.format(i, format_float(t0_stats["times"][i]),format_float(t1_stats["times"][i]))
    print(line)
  line = '{:>20}: {:>9} {:>9}'.format("StdDev", format_float(t0_stats["stddev"]),format_float(t1_stats["stddev"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("MAD", format_float(t0_stats["mad"]),format_float(t1_stats["mad"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("RMAD", format_float(t0_stats["rmad"]),format_float(t1_stats["rmad"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("Median", format_float(t0_stats["median"]),format_float(t1_stats["median"]))
  print(line)
  line = '{:>20}: {:>9} {:>9}'.format("Average", format_float(t0_stats["avg"]),format_float(t1_stats["avg"]))
  print(line)

if verbose:
  print("Num cases: ", len(bads)+len(wins)+len(neutrals))
  print("Num wins (dt <= -%.2f%%): "%threshold, len(wins))
  print("Num fails (dt >= +%.2f%%): "%threshold, len(bads))

if len(bads) > 0:
  print()
  print("Fail cases, where median time increased >= %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0_stats,t1_stats in bads:
    gain = '+%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>6}) (new: {:>6})'.format(format_bytes(size),dtype,placeness,redop,gain,format_float(t0_stats["median"]),format_float(t1_stats["median"]))
    print(line)
    print_stats(t0_stats, t1_stats)
  if exit_code == 0:
    exit_code = 1

if len(wins) > 0:
  print()
  print("Win cases, where median time decreased >= %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0_stats,t1_stats in wins:
    gain = '%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>6}) (new: {:>6})'.format(format_bytes(size),dtype,placeness,redop,gain,format_float(t0_stats["median"]),format_float(t1_stats["median"]))
    print(line)
    if verbose:
      print_stats(t0_stats, t1_stats)

if len(neutrals) > 0 and verbose:
  print()
  print("Neutral cases, where median time stayed +- %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0_stats,t1_stats in neutrals:
    gain = '%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>6}) (new: {:>6})'.format(format_bytes(size),dtype,placeness,redop,gain,format_float(t0_stats["median"]),format_float(t1_stats["median"]))
    print(line)
    print_stats(t0_stats, t1_stats)

if exit_code == 0:
  print("SUCCESS")
else:
  print("FAILURE")
exit(exit_code)
