import requests

r = requests.get('https://api.github.com/repos/jay-sharmaa/Fitlite/commits')

text = r.json()

additions = 0
deletions = 0

for i in range(0, len(text)):
    curr_sha = text[i]['sha']
    r = requests.get(f'https://api.github.com/repos/jay-sharmaa/Fitlite/commits/{curr_sha}')
    additions += r.json()['stats']['additions']
    deletions += r.json()['stats']['deletions']

print(additions)
print(deletions)
