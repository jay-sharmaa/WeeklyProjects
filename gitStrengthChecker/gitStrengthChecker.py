import requests

r = requests.get('https://api.github.com/repos/jay-sharmaa/Fitlite/commits')

text = r.json()

size = len(text)

for i in range(0, size):
    curr_sha = text[i]['sha']
    r = requests.get('https://api.github.com/repos/jay-sharmaa/Fitlite/commits/{curr_sha}')
print(text)