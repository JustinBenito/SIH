import requests

data = {
    "age": 12,
    "gender": "Male",
    "coursecount": 5,
    "timespent": 1000,
    "loginstreak": 1000,
    "score": 1000,
    "codingsolved": 1000,
    "skilllevel": "Intermediate"
}

response = requests.post("http://127.0.0.1:8000/predict/", json=data)
print(response.json())