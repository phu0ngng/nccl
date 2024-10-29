import json
from datetime import datetime, timezone, timedelta

gitlabTzInfo = timezone(timedelta(hours=-7))

# Specify the path to your JSON file
json_file_path = 'pipelines.json'

# Get the current datetime minus one day to see all jobs completed in the last day
cutoff = datetime.now(gitlabTzInfo) - timedelta(days=1)
try:
    # Open and read the JSON file of past pipelines
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Print header
    print("{:<40} {:<12} {:<16} {:<20} {:<10}".format("Pipeline", "ID", "Elapsed", "Created", "Status"))

    pipeline_count = 0
    succeeded_pipeline_count = 0
    min_time = datetime.now(gitlabTzInfo)
    sum_pipeline_time = timedelta(0)
    completed_pipeline_count = 0

    for item in json_data:
        created = datetime.fromisoformat(item["created_at"])
        updated = datetime.fromisoformat(item["updated_at"])
        status = item["status"]
        elapsed = updated - created
        if elapsed != timedelta(0) and cutoff <= updated:
            pipeline_count += 1
            if status == "success":
                succeeded_pipeline_count += 1
            if status != "canceled":
                sum_pipeline_time += elapsed
                completed_pipeline_count += 1
            if min_time > created:
                min_time = created
            print("{:<40} {:<12} {:<16} {:<20} {:<10}".format(item["ref"], item["id"], str(elapsed), created.strftime("%d-%m-%Y %H:%M:%S"), status))

    overall_elapsed = datetime.now(gitlabTzInfo) - min_time 
    print("\n{0} pipelines, {1} completed, {2} passed in the last day. {3} average completed time. {4} elapsed.\n".format(pipeline_count, completed_pipeline_count, succeeded_pipeline_count, sum_pipeline_time/completed_pipeline_count, overall_elapsed))

except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found.")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
