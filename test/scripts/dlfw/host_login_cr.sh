#!/bin/bash
attempt=0
max_attempts=5
while [ $attempt -le $max_attempts ]; do
    if echo "$CI_GITLAB_CR_ACCESS_TOKEN" | docker login gitlab-master.nvidia.com -u "/$oauth" --password-stdin; then
        echo "Command succeeded"
        break
    else
        echo "Attempt $attempt failed. Retrying..."
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "Command failed after $max_attempts attempts"
fi
