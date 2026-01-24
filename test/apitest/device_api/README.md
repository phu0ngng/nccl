# Overview

This directory contains apitests for the Device API.

In general, apitests have the following properties:
- single process, single thread
- one comm per visible device on the host

Most of the tests inherit functionality of the ncclDevApiCommon class.  This class:
- initializes comms at the beginning of every test suite and destroys them at the end of the test suite.
- provides a createDevComms helper function that creates devComms according to req. It is the responsibility of each test to call this function. This returns testSkip if the device API is not supported or if gin is requested but not supported
- destroys all devComms at the end of every test.