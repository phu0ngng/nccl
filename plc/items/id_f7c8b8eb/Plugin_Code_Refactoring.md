# Plugin Code Refactoring
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
NCCL allows older plugin versions to work through a backward compatibility layer. Every plugin (net/tuner/profiler)
has its own compat layer implementation (located in the net.cc/tuner.cc/profiler.cc, respectively). The compat layers
use a naming scheme of the form: function\_vX\_as\_vY, where X < Y and Y is the most recent plugin version supported by
NCCL. Every time we increment Y all the function names in the compat layer need to be updated. This often involves copy
and paste of old code and can cause errors to go undetected until later on a bug is reported.

The main problem with the current design is having compatibility layers and plugin loading in the same file, which
requires the inconvinient naming scheme to keep all the compatibility layers names distinct. Moreover, the compat
name is also used to identify the version of the plugin selected (encoded in the X).

<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->
 
### NVbugs / Jira Tickets
[Generate NCCL Plugin Code Automatically](https://nvbugspro.nvidia.com/bug/4912557)

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
Instead of having all the compatibility layer code in the same file as the plugin load/init code, place
compatibility code into separate files named net\_vX.cc and move those files to a separate plugin dir
under the main source directory tree: src/plugin/net. The original net.cc file, e.g., is moved to src/plugin,
along the other plugin files (i.e., tuner.cc and profiler.cc).

The directory tree with the respective plugin files is reported below

```
nccl
+-src
  +-include
  | +-plugin
  |   +-{nccl_net.h,nccl_tuner.h,nccl_profiler.h}
  |   +-net
  |   | +-{net_v5.h,net_v6.h,net_v7.h,net_v8.h,net_v9.h}
  |   +-tuner
  |   | +-{tuner_v2.h,tuner_v3.h}
  |   +-profiler
  |     +-{profiler_v1.h,profiler_v2.h}
  +-plugin
    +-{net.cc,tuner.cc,profiler.cc}
    +-net
    | +-{net_v5.cc,net_v6.cc,net_v7.cc,net_v8.cc,net_v9.cc}
    +-tuner
    | +-{tuner_v2.cc,tuner_v3.cc}
    +-profiler
      +-{profiler_v1.cc,profiler_v2.cc} 
```

The compatibility layer for, e.g., the network plugin is now located in src/plugin/net. The network plugin
loading code, in src/plugin/net.cc, calls getNcclNet\_vX() in src/plugin/net/net\_vX.cc to gets back a
populated symbol table (with the appropriate compat conversions) for the detected network plugin version.
Therefore, the src/plugin/net/net\_vX.cc files can drop the inconvinient naming scheme described above. Now,
if the version of the plugin is incremeneted there is no need to update all the compatibility layers
accordingly. If the update introduces any API changes, the developer is notified about stale compat layer
interfaces during build time.

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
[Refactor plugin handling code](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/734) 
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
  - Giuseppe Congiu <gcongiu@nvidia.com>

</details>
