import requests
response = requests.post("http://localhost:8000/topic/invoke", 
                         json={"input":{"topic":"generative ai "}})

print(response.json()['output']['content'])