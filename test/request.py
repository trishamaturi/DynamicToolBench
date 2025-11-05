import requests
import json
import os

url = 'http://0.0.0.0:8080/virtual'
data = {
    "category": "Artificial_Intelligence_Machine_Learning",
    "tool_name": "TTSKraken",
    "api_name": "List Languages",
    "tool_input": '{}',
    "strip": "truncate",
    "toolbench_key": ""
}
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.text)