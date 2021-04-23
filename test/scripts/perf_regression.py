#!/usr/bin/env python3
import os
import sys
import time
import threading

exes = [
  'all_gather_perf',
  'all_reduce_perf',
  'alltoall_perf',
  'broadcast_perf',
  'reduce_perf',
  'reduce_scatter_perf',
  'sendrecv_perf'
]

time_begin = time.time()
exit_code = 0

def env(x, deft=None):
  s = os.environ.get(x, deft)
  return s if deft is None else type(deft)(s)

def subproc(args, env=None):
  import subprocess as sp
  import shlex
  args = list(map(str, args))
  env = {x:str(y) for x,y in (env or os.environ).items()}
  p = sp.Popen(args, stdout=sp.PIPE, env=env)
  try:
    out, err = p.communicate(timeout=5*60)
  except sp.TimeoutExpired:
    exit_code = 1
    print('** FAILED ** command (timeout > 5min):\n' + ' '.join(map(shlex.quote, args)), file=sys.stderr)
    return None

  if p.returncode != 0:
    exit_code = p.returncode
    print('** FAILED ** command (return=%d):\n'%p.returncode + ' '.join(map(shlex.quote, args)), file=sys.stderr)
    return None
  else:
    return out.decode('utf-8')

LD_LIBRARY_PATH = env('LD_LIBRARY_PATH', '')
SALLOC = env('SALLOC', '')
MPIRUN = env('MPIRUN', 'mpirun -q --oversubscribe')
MPI_PROCS = env('MPI_PROCS', env('SLURM_NTASKS', 2))
MPI_NODES = env('MPI_NODES', env('SLURM_NNODES', 1))

def die():
  print("Usage: perf_regression.py old=<build-dir> new=<build-dir> threshold=<percent, default=5> <args-to-perf-exe>...")
  exit(1)

if '-h' in sys.argv or '--help' in sys.argv[1:]:
  die()

kwnames = ['new','old','threshold']
kwargs = {x[:x.index('=')]:x[x.index('=')+1:] for x in sys.argv[1:] if '=' in x and x[:x.index('=')] in kwnames}
exe_args = [x for x in sys.argv[1:] if '=' not in x or x[:x.index('=')] not in kwnames]

if 'new' not in kwargs: die()
build_new = kwargs['new']
if 'old' not in kwargs: die()
build_old = kwargs['old']
threshold = float(kwargs.get('threshold', 5))

def run_perf_mpi(args, env):
  xenv = []
  for x,y in env.items():
    if x.startswith('NCCL_') or x in ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'):
      xenv += ['-x','%s=%s'%(x,y)]
  import shlex
  salloc = shlex.split(SALLOC)
  mpirun = shlex.split(MPIRUN)
  return subproc(salloc + mpirun + xenv + args, env)

def run_perf(topo, args, env, key_prefix, times):
  if topo == 'intra_proc':
    mpiargs = ['-N',1, '-np',MPI_NODES]
    args = args + ['-g',MPI_PROCS//MPI_NODES]
  elif topo == 'inter_proc':
    mpiargs = ['-N',MPI_PROCS//MPI_NODES, '-np',MPI_PROCS]
    args = args + ['-g',1]
  else:
    assert 0

  out = run_perf_mpi(mpiargs + args + ['--out_of_place',0], env)

  for ln in (out or '').split('\n'):
    ln = ln.rstrip()
    if ln.startswith('#'):
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
        t = float(field(ln,9)) # in-place time
        key = key_prefix + (size,dtype,redop)
        times[key] = min(t, times.get(key, t))
      except:
        print('BAD line:'+ln)
        raise

def sweep(times):
  env = dict(os.environ) # copy of env
  def csv(s):
    return [x.strip().lower() for x in s.split(',') if x!='']
  filter_exes = csv(env.get('NCCL_EXES',''))
  filter_protos = csv(env.get('NCCL_PROTOS',''))
  filter_algos = csv(env.get('NCCL_ALGOS',''))
  filter_topos = csv(env.get('NCCL_PROC_TOPOS',''))
  proc_topos = [x for x in ['intra_proc','inter_proc'] if not filter_topos or x in filter_topos]
  for trial in range(1):
    for topo in proc_topos:
      for exe in [x for x in exes if not filter_exes or x.lower() in filter_exes]:
        protos = ['LL','LL128','SIMPLE']
        algos = ['RING','TREE']
        if exe in ['alltoall_perf','sendrecv_perf']:
          protos = ['SIMPLE']
          algos = ['RING']
        elif exe in ['all_gather_perf','broadcast_perf','reduce_perf','reduce_scatter_perf']:
          algos = ['RING']

        if exe in ['all_reduce_perf','reduce_perf','reduce_scatter_perf']:
          ops = [['-o','all', '-d','all']]
        else:
          ops = [['-d','int8']]

        if topo == 'intra_proc':
          protos = ['SIMPLE']
          algos = ['RING']

        if topo != 'inter_proc':
          ops = [['-o','min', '-d','int8'],
                 ['-o','max', '-d','half'],
                 ['-o','sum', '-d','float'],
                 ['-o','sum', '-d','int64']]
        #protos = ['LL']
        #algos = ['TREE']
        #ops = ['-d','int8']

        for proto in [x for x in protos if not filter_protos or x.lower() in filter_protos]:
          sizes = {
            'LL':     ['-b64',  '-e64K', '-f1024', '-n500'],
            'LL128':  ['-b256K','-e256M','-f32'],
            'SIMPLE': ['-b32M', '-e1G',  '-f32']
          }[proto]
          for algo in [x for x in algos if not filter_algos or x.lower() in filter_algos]:
            for op in ops:
              env['NCCL_PROTO'] = proto
              env['NCCL_ALGO'] = algo
              # run NEW test with old libnccl
              print('Running part=%s trial=%d proto=%s algo=%s build=%s %s'%(topo,trial,proto,algo,build_old,' '.join(map(str,[exe]+sizes+op+exe_args))))
              env['LD_LIBRARY_PATH'] = build_old+'/lib:'+LD_LIBRARY_PATH
              run_perf(topo, [build_new+'/test/perf/'+exe] + sizes + op + exe_args, env, ('old',exe,topo,proto,algo), times)
              # run new test with new libnccl
              print('Running part=%s trial=%d proto=%s algo=%s build=%s %s'%(topo,trial,proto,algo,build_new,' '.join(map(str,[exe]+sizes+op+exe_args))))
              env['LD_LIBRARY_PATH'] = build_new+'/lib:'+LD_LIBRARY_PATH
              run_perf(topo, [build_new+'/test/perf/'+exe] + sizes + op + exe_args, env, ('new',exe,topo,proto,algo), times)

times = {}
sweep(times)

bads=[]
wins=[]
neutrals = 0
for (oldnew,exe,topo,proto,algo,size,dtype,redop),t0 in times.items():
  if oldnew == 'old':
    try:
      t1 = times['new',exe,topo,proto,algo,size,dtype,redop]
      gain = 100*(t1 - t0)/t0
      if gain >= threshold:
        bads += [(gain,exe,topo,proto,algo,size,dtype,redop)]
      elif gain <= -10.0:
        wins += [(gain,exe,topo,proto,algo,size,dtype,redop)]
      else:
        neutrals += 1
    except KeyError:
      bads += [('FAILED',exe,topo,proto,algo,size,dtype,redop)]
bads.sort(key=lambda x:(str(type(x)),x), reverse=True)
wins.sort(key=lambda x:(str(type(x)),x), reverse=False)

time_end = time.time()

print("Elapsed seconds : %.2f"%(time_end-time_begin))

def format_bytes(n):
  if n < 1<<10:
    return "%gB"%n
  elif n < 1<<20:
    return "%.3gK"%(n/(1<<10))
  elif n < 1<<30:
    return "%.3gM"%(n/(1<<20))
  else:
    return "%.3gG"%(n/(1<<30))

print("Num cases: ", len(bads)+len(wins)+neutrals)
print("Num wins (dt <= -10%): ", len(wins))
print("Num fails (dt >= +%.2f%%): "%threshold, len(bads))

if len(bads) > 0:
  print()
  print("Fail cases, where time increased >= %.2f%%:"%threshold)
  for gain,exe,topo,proto,algo,size,dtype,redop in bads:
    gain = '+%.2f%%'%gain if type(gain) in (int,float) else gain
    print('  %s %s %s %s %s %s %s : %s'%(exe,topo,proto,algo,format_bytes(size),dtype,redop,gain))
  if exit_code == 0:
    exit_code = 1

if len(wins) > 0:
  print()
  print("Win cases, where time decreased >= 10%:")
  for gain,exe,topo,proto,algo,size,dtype,redop in wins:
    gain = '%.2f%%'%gain if type(gain) in (int,float) else gain
    print('  %s %s %s %s %s %s %s : %s'%(exe,topo,proto,algo,format_bytes(size),dtype,redop,gain))

if exit_code == 0:
  print("SUCCESS")
else:
  print("FAILURE")
exit(exit_code)
