# Use NCCL tools containers to build and test NCCL

To use the tools container, execute `docker/make.sh`

* If `--clean` parameter is passed, previous *build* directory will be removed
* The default target cluster name is inferred from the build host name
* If a cluster name is passed as a parameter (currently supported list is: `gc`, `eos`, `draco-rno`, `draco-oci`, and `all`), it will be used as the target cluster name and the corresponding gencode will be passed to *make*
* *make* will be executed inside the build tools container with source mounted in */nccl* path

