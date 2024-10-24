#!/bin/bash

echo "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs"
curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs" > data.json
python3 test/scripts/parse_summary.py
