# Support multiple net plugins
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
NCCL_NET_PLUGIN variable implementation has been modified to allow a comma separated list
of plugins instead of only one plugin. NCCL will go through the list until it finds a plugin
that successfully loads and initializes.
<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
The current NCCL implementation makes it impossible to release a NCCL container that
can work on different networks. With a list of plugins assigned to NCCL_NET_PLUGIN,
we can add more than one plugin in a container to support different networks. NCCL can
then go through the provided list and initialize the appropriate plugin.
<!-- ============================================================================================-->
 
### NVbugs / Jira Tickets
NvBug: [4578372](https://nvbugspro.nvidia.com/bug/4578372)

### User Experience
Current usage of NCCL_NET_PLUGIN can remain unchanged. In addition to this, users
can now assign a list of plugins.

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
In the current implementation, net plugin load code loads only one external plugin and then tries
to initialize the external plugin. If that fails, it will try to initialize internal IB and
socket plugin. 
With the new design we are expanding the externally provided plugins to 14. (16 - 2 for internal IB and socket plugins)
For each povided plugin, we will maintain its state and other parameters such as ref count and handle.
This design will help us in future as well when we want to enable multiple plugins simultaneously. 
NCCL net init goes through the list of plugins. It loads and tries to initialize the plugin, if load or init fails, it
tries the next plugin in the list. At the end of the list, it attempts to load IB and socket plugin.
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
Plugin load and init will happen one after another instead of trying to load plugins first and initalize later.
net.cc contains many global arrays: `ncclNets, ncclCollNets, ncclNetsVer, ncclNetStates, ncclCollNetState`
This MR encapsulates all of them into one struct netPluginLib. Array of this struct will be a global entity.
We will pass pluginIndex to plugin load and init functions and they will deal with the appropriate element
in the array.
<!-- ============================================================================================-->

### Commit list or MR
 https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/761
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
  - Unmesh Deodhar

</details>
