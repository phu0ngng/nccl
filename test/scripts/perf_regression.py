#!/usr/bin/env python3
import sys

def help():
  print("Usage: perf_regression.py old=<baseline_perf_file> new=<new_perf_file> threshold=<percent, default=10> verbose=<True/False, default=False>")
  exit(1)

if '-h' in sys.argv or '--help' in sys.argv[1:]:
  help()

kwnames = ['new','old','threshold','verbose']
kwargs = {x[:x.index('=')]:x[x.index('=')+1:] for x in sys.argv[1:] if '=' in x and x[:x.index('=')] in kwnames}

if 'new' not in kwargs: help()
file_new = kwargs['new']
if 'old' not in kwargs: help()
file_old = kwargs['old']
threshold = float(kwargs.get('threshold', 10))
verbose = float(kwargs.get('verbose', False))
exit_code = 0

def compare_times(placeness, bads, wins, neutrals, new_times, old_times):
  for (size,dtype,redop),t0 in old_times.items():
    try:
      t1 = new_times[size,dtype,redop]
      gain = 100*(t1 - t0)/t0
      if gain >= threshold:
        bads += [(placeness, gain,size,dtype,redop,t0,t1)]
      elif gain <= -threshold:
        wins += [(placeness, gain,size,dtype,redop,t0,t1)]
      else:
        neutrals += [(placeness, gain,size,dtype,redop,t0,t1)]
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
        oop_times[key] = float(field(ln,5)) # out-of-place time
        inp_times[key] = float(field(ln,9)) # in-place time
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
compare_times("out-of-place", bads, wins, neutrals, new_oop_times, old_oop_times)
compare_times("in-place", bads, wins, neutrals, new_inp_times, old_inp_times)

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

print("Num cases: ", len(bads)+len(wins)+len(neutrals))
print("Num wins (dt <= -%.2f%%): "%threshold, len(wins))
print("Num fails (dt >= +%.2f%%): "%threshold, len(bads))

if len(bads) > 0:
  print()
  print("Fail cases, where time increased >= %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0,t1 in bads:
    gain = '+%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>12}) (new: {:>12})'.format(format_bytes(size),dtype,placeness,redop,gain,t0,t1)
    print(line)
  if exit_code == 0:
    exit_code = 1

if len(wins) > 0:
  print()
  print("Win cases, where time decreased >= %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0,t1 in wins:
    gain = '%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>12}) (new: {:>12})'.format(format_bytes(size),dtype,placeness,redop,gain,t0,t1)
    print(line)

if len(neutrals) > 0 and verbose:
  print()
  print("Neutral cases, where time stayed +- %.2f%%:"%threshold)
  for placeness,gain,size,dtype,redop,t0,t1 in neutrals:
    gain = '%.2f%%'%gain if type(gain) in (int,float) else gain
    line = '{:>6} {:>6}  {:>12} {:>6} : {:>6} (old: {:>12}) (new: {:>12})'.format(format_bytes(size),dtype,placeness,redop,gain,t0,t1)
    print(line)

if exit_code == 0:
  print("SUCCESS")
else:
  print("FAILURE")
exit(exit_code)
