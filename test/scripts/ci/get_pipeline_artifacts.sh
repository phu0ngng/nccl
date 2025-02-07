#!/bin/bash

# GitLab API URL
GITLAB_URL="https://gitlab-master.nvidia.com/api/v4"

echo "Your GitLab personal access token should be stored as \$GITLAB_API_TOKEN from https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens"

# Project ID (you can find this in your project's settings)
PROJECT_ID="34033"

# Pipeline ID (you can get this from the pipeline URL or via API)
REF_NAME="$1"

prompt_yes_no() {
    while true; do
        read -p "$1 [y/n]: " response
        case $response in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

if prompt_yes_no "Delete existing artifacts?"; then
    rm -rf failed/
    rm -rf junit*
    rm -rf build*
    rm -rf run*
fi

# Function to download artifacts for a job
download_artifacts() {
    local job_id=$1
    local job_name=$2
    echo "Downloading artifacts for job: $job_name (ID: $job_id)"
    curl --location --output "${job_name}_artifacts.zip" \
         --header "PRIVATE-TOKEN: $TOKEN" \
         "$GITLAB_URL/projects/$PROJECT_ID/jobs/$job_id/artifacts"
    unzip -o ${job_name}_artifacts.zip
    mkdir -p junits/$job_name
    mv *.xml junits/$job_name
    rm ${job_name}_artifacts.zip
}

# Function to get jobs from a specific page
get_jobs_page() {
    local page=$1
    curl --silent --header "PRIVATE-TOKEN: $TOKEN" \
         "$GITLAB_URL/projects/$PROJECT_ID/pipelines/$PIPELINE_ID/jobs?per_page=100&page=$page"
}

# Function to check if job has valid artifacts
has_valid_artifacts() {
    local artifacts="$1"
    echo "$artifacts" | jq -e '[.[] | select(.filename != "job.log" and .filename != "metadata.gz")] | length > 0' > /dev/null
}

get_pipeline_id() {
    # Ref name from command line argument
    REF_NAME="$1"

    # Make the API request and get the latest pipeline ID
    LATEST_PIPELINE_ID=$(curl -s --header "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
        "$GITLAB_URL/projects/$PROJECT_ID/pipelines?ref=$REF_NAME&order_by=updated_at&sort=desc&per_page=1" \
        | jq '.[0].id')

    # Check if a pipeline ID was found
    if [ -z "$LATEST_PIPELINE_ID" ] || [ "$LATEST_PIPELINE_ID" == "null" ]; then
        # REF_NAME is pipeline ID
        echo "$REF_NAME"
    else
        echo "$LATEST_PIPELINE_ID"
    fi
}

# Initialize variables
let page=1
let total_pages=1

set +x
PIPELINE_ID=$(get_pipeline_id $REF_NAME)
echo "PIPELINE_ID=$PIPELINE_ID"

# Loop through all pages
while [[ $page -le $total_pages ]]; do
    # Get jobs for the current page
    response=$(get_jobs_page $page)
    # Extract total pages from the response headers if it's the first page
    if [ $page -eq 1 ]; then
        total_pages=$(echo "$response" | grep -i "x-total-pages:" | awk '{print $2}' | tr -d '\r')
    fi

    # Process each job on the current page
    echo "$response" | jq -c '.[]' | while read job; do
        job_id=$(echo $job | jq -r '.id')
        job_name=$(echo $job | jq -r '.name')
        artifacts=$(echo $job | jq -r '.artifacts')
        if has_valid_artifacts "$artifacts"; then
            download_artifacts $job_id $job_name
        else
            echo "No valid artifacts found for job: $job_name (ID: $job_id)"
        fi
    done

    # Move to the next page
    page=$((page + 1))
done

echo "Artifact download complete!"
echo "Found xmls:"
find junits -name "*.xml"
echo "Found repro scripts:"
find failed -name "repro.sh"
