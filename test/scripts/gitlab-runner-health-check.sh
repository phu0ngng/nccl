#!/bin/bash -e
nvidia-smi --format=csv,noheader --query-gpu=persistence_mode | grep Enabled

