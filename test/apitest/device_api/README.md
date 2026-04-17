# Overview

This directory contains apitests for the Device API.

In general, apitests have the following properties:
- single process, single thread
- one comm per visible device on the host

Most of the tests inherit functionality of the `ncclDevApiCommon` class.  This class:

* initializes comms at the beginning of every test suite and destroys them at the end of the test suite.
* provides a createDevComms helper function that creates devComms according to req. It is the responsibility of each test to call this function. This returns testSkip if the device API is not supported or if gin is requested but not supported
* destroys all devComms at the end of every test.

There is also a `ncclMultiTeamCommon` class which provides the same functionality as `ncclDevApiCommon` and also artifically decreases the LSA team size to 2 so we can test multi-team logic on a single host. Prefer `ncclDevApiCommon` unless your test absolutely needs multiple teams.

The **reduceCopy** subdirectory contains parameterized tests for the ReduceCopy device API. Those tests use their own fixture (`ReduceCopyTestBase`) and skip per test when device API is not supported (`TestSupportChecker::isDeviceApiSupported`). ReduceCopy test types and generation are controlled by `NCCL_TEST_REDUCE_COPY_TYPES` (see reduceCopy/generate_tests.py).
