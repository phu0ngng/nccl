# Use NCCL tools containers to build and test NCCL

To use the build tools container, execute `docker/make.sh`

* If `--clean` parameter is passed, previous *build* directory will be removed
* The default target cluster name is inferred from the build host name
* If a cluster name is passed as a parameter (currently supported list is: `gc`, `gc-classic`, `eos`, `draco-rno`, `draco-oci`), it will be used as the target cluster name and the corresponding gencode will be passed to *make*. Note that `gc-classic` uses `cuda-11.7`-based build container.
* If a comma-separated list of cluster names is used, the build will include all the gencodes for each cluster and will use build container setting for the last cluster on the list
* *make* will be executed inside the build tools container with source mounted in */nccl* path

To use run tools container, please see `docker/examples` directory.

