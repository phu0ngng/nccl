#!/bin/bash

# For debugging
# export CI_JOB_TOKEN=""
# export CI_PIPELINE_ID=19830744
# export CI_API_V4_URL="https://gitlab-master.nvidia.com/api/v4/"
# export CI_PROJECT_ID=34033

echo "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines?order_by=updated_at&scope=finished"
curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines?order_by=updated_at&scope=finished" > pipelines.json
python3 test/scripts/parse_finished_pipelines.py

echo "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs"
curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/pipelines/${CI_PIPELINE_ID}/jobs" > jobs.json
python3 test/scripts/parse_pipeline_jobs.py
