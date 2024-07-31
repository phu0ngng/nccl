# Use NCCL tools containers to build and test NCCL

To use the tools container, execute *docker/make.sh* with an optional cluster name.

* If `--clean` parameter is passed, previous *build* directory will be removed
* *make* will be executed inside the build tools container with source mounted in */nccl* path

