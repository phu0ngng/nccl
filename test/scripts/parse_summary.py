import json
from datetime import datetime

# Specify the path to your JSON file
json_file_path = 'data.json'

try:
    # Open and read the JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
    
    # Print the parsed JSON data
    # print(json.dumps(json_data, indent=2))


    # Print header
    print("{:<30} {:<20} {:<20}".format("Job", "Waiting", "Running"))

    for item in json_data:
        created = datetime.fromisoformat(item["created_at"])
        if item["started_at"] is None:
            running = "Pending"
            started = datetime.datetime.now()
        if item["finished_at"] is None:
            running = "Running"
        else:
            finished = datetime.fromisoformat(item["finished_at"])
            running = str(finished - started)
        waiting = str(started - created)
        print("{:<30} {:<20} {:<20}".format(item["name"], waiting, running))
except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found.")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
