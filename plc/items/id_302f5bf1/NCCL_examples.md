# NCCL examples
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract

The NCCL examples folder provides users and developers with practical code samples that highlight NCCL’s core features. It covers a range of cases, from basic operations like communicator initialization, point-to-point communication, and collective operations, to advanced features such as User Buffer (UB), symmetric memory, and the device API.

The set of tests covered in this PLC is a minimal subset of available NCCL APIs to get started. Future contributions will be added based on user feedback and demand.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
The examples are categorized into "basic examples" and "advanced features". To learn the general use of NCCL new users are guided through self-contained examples highlighting all necessary steps to create a running NCCL application. 
For users exploring specific features we show each "advanced feature" separately for easier readability.

### NVbugs / Jira Tickets

- NVBugs: https://nvbugspro.nvidia.com/bug/5478088
- GitHub: https://github.com/NVIDIA/nccl/issues/1818

### User Experience

- This examples directory will be part of the NCCL GitHub repo, but not be tied to the NCCL release cycle.
- A new top level Makefile target has been created such that `make examples` builds the NCCL library, then all examples. Each subdirectory has it's own makefile so user can choose to build just a subset or individual example. The examples are not built by default.
- The individual executables run independently and only require `libnccl.so` as additional dependency. The MPI based examples need to be provided a `MPI_HOME`.

The new user can start with the most basic example `01_communicator/01_multiple_devices_single_process`. From there they can either look at adding pthreads (or MPI) in `01_communicator/02_one_device_per_pthread` (`01_communicator/03_one_device_per_process_mpi`) *or* add more NCCL functionality in `02_point_to_point/01_ring_pattern` and `03_collectives/01_allreduce`.

The more advanced user can directly look at an example for an individual feature and will find a function which includes all the necessary NCCL API calls.

### Assumptions, constraints and dependencies

- We assume that the user uses the same environment as they us to build NCCL. Ideally calling `cd nccl; make examples`.
- The dependencies are the same as for `nccl/src`.
- By default the examples will run on a single node using one thread per GPU. Users can enable MPI when they provide an external `MPI_HOME`.

### Use Cases

- Users want to get started building a NCCL application or quickly run a functional NCCL test.
- Users want an integrated version of the examples given in the User Docs.
- Users want to quickly see best practices using new features.

### Platform Requirements

- This examples directory only needs to support features in the current release. This means we do not have to check for backwards compatibility, but make sure we update the examples if we add breaking change into any NCCL API.

<!-- ### Functional Requirements -->
<!-- ### System Requirements -->
<!-- ### Interface Requirements -->
<!-- ### KPI Requirements -->
<!-- ### Security Requirements -->
<!-- ### Legal and Standards Requirements -->
<!-- ### Telemetry Requirements -->
<!-- ### Backward Compatibility Requirements -->
<!-- ### Virtualization Requirements -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design
```
examples
├── 01_communicators
│   ├── 01_multiple_devices_single_process
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── 02_one_device_per_pthread
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── 03_one_device_per_process_mpi
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── 02_point_to_point
│   ├── 01_ring_pattern
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── 03_collectives
│   ├── 01_allreduce
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── 04_user_buffer_registration
│   ├── 01_allreduce
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── 05_symmetric_memory
│   ├── 01_allreduce
│   │   ├── main.cc
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── 06_device_api
│   ├── 01_allreduce
│   │   ├── main.cu
│   │   ├── Makefile
│   │   └── README.md
│   ├── Makefile
│   └── README.md
├── common
│   ├── include
│   │   ├── mpi_utils.h
│   │   ├── nccl_utils.h
│   │   └── utils.h
│   ├── README.md
│   └── src
│       └── utils.cc
├── compile_commands.json
├── Makefile
└── README.md
```

## Implementation Plan

### Step 1: Create Directory Structure
- Add the examples/ directory to the NCCL repository.
- Populate it with subdirectories (common, basic, advanced, profiling) as described above.
- Add Makefile into each directory such that users are able to build from each level.
- Add README.md which provides instructions for building and running the examples, include links to relevant NCCL documentation.

### Step 2: Develop Common Utilities
- Implement shared utilities in common/include/nccl_utils.h and common/src/utils.cpp.
- Include functions for NCCL initialization, error checking, and resource cleanup.

### Step 3: Write Basic Examples
- Implement simple examples for basic NCCL operations (all_reduce, broadcast, etc.).
- Ensure each example is self-contained and includes comments explaining the code.

### Step 4: Write Advanced Examples
- Develop examples for advanced NCCL features, such as multi-node communication and custom communicators.
- Include detailed documentation for these examples.

<!-- [Example](images/example.png) -->
<!-- note: the following HTML code is also valid -->
<!-- <img src="images/example.png" width="900" height="500" /> -->

### Interface Architecture

<!-- ### System KPIs & Metrics -->
<!-- ### Data Architecture -->
<!-- ### Security Design -->
<!-- ### Debugging & Troubleshooting -->
<!-- ### Logging and Instrumentation -->
<!-- ### Operational Considerations -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1171

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

Examples are built and run as part of the CI pipeline.

#### Where to run?

#### What to run?

#### Expected output?

<!-- #### Code Coverage Goal Defined -->
<!-- #### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency) -->
<!-- #### Requirement Coverage Goal Defined -->
<!-- #### Quality Thresholds Defined -->
<!-- #### Test Timeline -->
<!-- #### SW Verification and Test Plan -->
<!-- ### Test plan -->
- The examples will undergo QA and be run in CI.
<!-- #### Requirements Tests -->
<!-- #### Interface Tests -->
<!-- #### Fault-injection Tests -->
<!-- #### Resource Usage Tests -->
<!-- #### Design Coverage Testing -->
<!-- #### Boundary Tests -->
<!-- #### Certification Tests -->
<!-- #### Stress Tests -->
<!-- #### Stability Tests -->
<!-- #### Perf and Power KPI Tests -->
<!-- #### Usability & OOBE Tests -->
<!-- #### Manufacturing Diagnostic(Factory) Tests -->

### Performance
None expected. Only functional tests.

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Wahid Mainassara
  - Stephen Sachs
</details>
