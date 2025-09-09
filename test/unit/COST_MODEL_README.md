# NCCL Cost Model Debugging and New Platform Onboarding

## Overview

The NCCL cost model is used to predict communication latency for different algorithms and protocols on any system. It models the latencies and bandwidths of various interconnects, CPUs, GPUs and algorithms.

To validate the cost model, we have the cost model tool. It's a host only tool that can quickly reproduce the cost model's estimates for any platform or scale.

## Directory Structure

```
topo/
├── {system name}/
│   ├── system.xml           # Hardware topology definition
│   └── data/
│       └── {ngpus_per_node}/
│           └── {nnodes}/
│               └── {function}/
│                   ├── time.txt           # Default perf
│                   ├── {algo}/
│                   │   ├── {proto}/
│                   │   │   └── time.txt   # Perf forcing this algo and proto
```

## 1. Onboarding a new platform

Each platform requires a `system.xml` file, collected with NCCL_TOPO_DUMP_FILE=system.xml.

## 2. Performance Data Collection

Measured performance data is stored in `time.txt` files with format:
```
# One latency value per line (microseconds)
125.4
126.1
124.8
...
```

Each line corresponds to the next power of two message size. time.txt files range from 8B to 4GB.

Collect a time.txt file using out of place latency for each split, scale, function, algo and proto you have available. Ensuring the data is run with good settings and is collected on a clean, performant fabric is crucial to be able to tune properly.

## 3. Add the platform as a default case in the model tool

Add your new platform to the list of default platforms to keep covering it in CI and future model changes. Add it to `const char* platforms[]`.

### Tuning Model Constants

The main cost model function is `ncclTopoTuneModel()`, defined in `src/graph/tuning.cc`. You can look at the default tuner constants under `ncclTunerConstants_t`.

## Output Interpretation

### Normal Mode
```
Platform/NodesxGPUs, Function
-----------+---------------------+-------------------------------+
     Size  |     Tree/Simple     |           Default            |
           |   data      model   |   best    dryrun      data   |
-----------+---------------------+-------------------------------+
         8 |   125.4     124.1   |  124.1    125.4      125.4   |
        16 |   126.2     125.8   |  125.8    126.2      126.2   |
```

### Compact Mode
```
  Platform/ Nodes Ngpus |    Delta at size 8 to 4G     | Score
-----------------------+-------------------------------+--------
      DGX-2V/    2x    8 | ###XXXXOOOOOOOOOOOOOOOOOOOOO | 98.5 %
```

**Legend:**
- (Blue): Model accuracy > 110% (model too optimistic)
- (Green): Model accuracy 95-110% (acceptable)
- (Yellow): Model accuracy 80-95% (model too pessimistic) 
- (Red): Model accuracy < 80% (significant error)

## Environment Variables

The model can be configured via environment variables:

```bash
export NCCL_MODEL_TEST_NGPUS=8        # Override GPU count
export NCCL_MODEL_TEST_NNodes=2       # Override node count
export NCCL_MODEL_TEST_Platform=DGX-2V # Override platform
export NCCL_MODEL_TEST_Function=AllGather # Override function
export NCCL_MODEL_TEST_CompactMode=1   # Enable compact mode
export NCCL_MODEL_TEST_DispMode=2      # Set display mode (busbw)
```

## Usage

### Basic Usage

```bash
# Test the default set of platforms
./model

# Test specific platform and configuration
./model -p DGX-2V -n 2 -g 8 -f AllReduce

# Compact mode for overview across configurations
./model -c 1
```

### Command Line Options

```bash
-n, --nnodes <number>     Number of nodes [default: 1 to 128]
-g, --ngpus <number>      GPUs per node [default: All available]
-p, --platform <name>     Platform to test [default: All platforms]
-f, --function <func>     NCCL function to test [default: AllReduce]
-c, --compact <0|1>       Compact display mode [default: auto]
-m, --mode <0|1|2>        Display mode: 0=time, 1=algbw, 2=busbw
-h, --help               Show help message
```
