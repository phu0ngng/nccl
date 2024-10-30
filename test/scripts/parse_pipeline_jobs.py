import json
from datetime import datetime, timezone, timedelta

# Specify the path to your JSON file
json_file_path = 'jobs.json'

try:
    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    
    gitlabTzInfo = timezone(timedelta(hours=-7))

    # Print header
    print("{:<30} {:<20} {:<20}".format("Job", "Waiting", "Running"))

    for item in json_data:
        created = datetime.fromisoformat(item["created_at"])
        started = datetime.now(gitlabTzInfo)
        running = "Pending"
        if item["started_at"] is not None:
            started = datetime.fromisoformat(item["started_at"])
            running = "Running"
        if item["finished_at"] is not None:
            finished = datetime.fromisoformat(item["finished_at"])
            running = str(finished - started)
        waiting = str(started - created)
        print("{:<30} {:<20} {:<20}".format(item["name"], waiting, running))
except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found.")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
