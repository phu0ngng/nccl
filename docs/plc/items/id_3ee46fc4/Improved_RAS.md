# Improved RAS
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

Enhancements to the NCCL RAS (Reliability, Availability, and
Serviceability) subsystem.  User-visible improvements include the support
for collecting and printing of per-collective-type operation counts and
for printing additional information about missing ranks in the communicator
output.  Internal improvements include a workaround for collective counters
going out-of-sync for some graph-captured operations, initial work towards
gracefully handling communicators with non-globally-unique commHashes,
refactoring of dynamic data structures to improve code robustness, and
more.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

RAS, first included in NCCL 2.24, suffers from limited functionality
compared to its original design, as well as from rough edges in several
areas.  For 2.26, we aim to address some of these rough edges:

* **Collective operation counters**: the initial implementation featured a
single counter per rank.  Leveraging the work on the profiling API,
separate counter per each collective operation type is now supported,
potentially making it possible to detect mismatched collectives.  This
effort also revealed a shortcoming of the CPU-side counting of collective
operations: it does not work reliably with graph-captured collectives,
potentially leading to the counters getting out-of-sync across ranks, and
thus to false-positive RAS warnings.  This is not easily solvable without a
major performance penalty, so for now we've applied a WAR of disabling the
counters for graph-captured collectives (unless the profiler is active).

* **Missing communicator ranks**: in the initial implementation, when
communicator data failed to include information from some of the ranks, the
error information being printed was limited to the missing rank numbers.
That's because the detailed information was self-reported by each rank, so
in the absence of data from a rank, there was nothing to print but the rank
number.  However, additional information on each rank _is_ available from
any rank that's responsive.  To avoid repeated queries (first a collective
query for data from all the ranks, then a follow-up query about the missing
ranks), the "missing" data is collected by the first encountered rank of
every communicator, and is subsequently replaced by the actual rank data as
the responses from the other ranks come in.  The collective data request --
propagated from rank to rank to collect the rank data -- includes the
expanding list of communicators that have already been encountered, so that
the "missing" data is not collected on every rank.

* **Non-unique commHash values**: RAS needs to be able to uniquely identify
each communicator that it keeps track of; in the initial implementation
this was done using the 64-bit `commHash` element of the NCCL communicator
structure.  However, these values are not guaranteed to be globally
unique and, before NCCL 2.24 release, it was in fact trivial to generate
communicators with duplicate `commHash` values using `ncclCommSplit`, by
repeatedly splitting the same communicator using an identical `color`
value.  Moving forward, we are working to simplify this unrealistic uniqueness
requirement by expanding it to _a tuple_ that, in addition to the commHash,
includes the host hash and the pid hash of the rank 0 process of the
communicator.  Also, a commHash will be required to be unique only within
each NCCL process, not globally.  The RAS
subsystem has already been updated, but the testing and enforcement of the
per-process commHash uniqueness in the core NCCL code is more difficult and
is planned for the next release.

* **Internal data structures**: day-to-day experience with writing and
debugging the RAS subsystem code has shown that some of the initial choices
made for internal data structures were suboptimal.  In particular, the use
of dynamically reallocated arrays of structures, once it was combined with
the fault-tolerance mechanisms, made the lifetime of such objects
unpredictable, resulting in dangling pointers in unexpected places.  The
realization came too late for a refactoring in the 2.24 time frame, but for
2.26 we replaced all such arrays with linked lists.  Additionally,
temporary data structures used for anomaly detection and output formatting
were also refactored; initially they were often full copies of various
input arrays; we realized that significant memory savings (and potentially
some performance improvements as well) could be realized by leveraging
pointers to the elements of the original arrays.

* **RAS termination**: in the initial implementation, RAS -- once started
during the first `ncclCommInit` -- would continue running even after the
last communicator was terminated.  This is still the case, but now RAS also
installs an `atexit` handler that releases all the resources and terminates
the RAS thread at process termination time, so that the resources are no
longer reported as "leaked" by 3rd party leak detection tools.  Terminating
RAS during the tear-down of the last communicator is something we would
like to revisit in the future; there are distributed data consistency
issues that make the problem nontrivial.

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5022959 [RFE NCCL] RAS Enhancements for NCCL 2.26

https://jirasw.nvidia.com/browse/NCCL-1729

### User Experience

### Assumptions, constraints and dependencies

### Use Cases

### Platform Requirements

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

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/769 (RAS improvements for 2.26)

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

### Validation

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

#### What is measured?

#### Results


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Kamil Iskra

</details>
