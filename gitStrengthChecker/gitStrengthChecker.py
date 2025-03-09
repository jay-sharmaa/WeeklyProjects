import requests
import re
import matplotlib.pyplot as plt

type = int(input('Give User Link/Name and Repository -- 0/1: '))

url = ''

if type == 0:
    user_name = input('Type User Name: ')
    repo_name = input('Type Repository Name: ')
    url = f'https://api.github.com/repos/{user_name}/{repo_name}/commits'
elif type == 1:
    url = input('Paste The Url Here: ')
    result = re.search(r'https://github.com/([^/]+)/([^/]+)/commits', url)
    url = f'https://api.github.com/repos/{result.group(1)}/{result.group(2)}/commits'
    print(url)

r = requests.get(url)

text = r.json()

print(text)

additions = []
deletions = []

addition = 0
deletion = 0

x_axis = []
for i in range(0, min(len(text), 12)):
    x_axis.append(i)

for i in range(0, min(len(text), 12)):
    curr_sha = text[i]['sha']
    
    r = requests.get(f'{url}/{curr_sha}')
    commit_details = r.json()

    for file in commit_details['files']:
        addition += file['additions']
        deletion += file['deletions']
        
        if 'patch' in file:
            print("Patch content for file:", file['filename'])
            print()
        
    additions.append(addition)
    deletions.append(deletion)

plt.plot(x_axis, additions, color='blue', label='additions')
plt.plot(x_axis, deletions, color='red', label = 'deletions')

plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')

plt.title('Project Relevance')

plt.grid(True)

plt.show()