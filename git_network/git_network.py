import networkx as nx
import requests
import matplotlib.pyplot as plt
import time

def get_followers(username, max_followers=5):
    url = f"https://api.github.com/users/{username}/followers"
    params = {'per_page': max_followers}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return [user['login'] for user in response.json()]
    else:
        print(f"Failed {response.status_code} for user {username}")
        return []

def build_follower(root_user, max_followers=5, max_depth=2):
    G = nx.DiGraph()
    visited = set()

    def dfs(user, depth):
        if depth > max_depth or user in visited:
            return
        visited.add(user)

        followers = get_followers(user, max_followers)
        for follower in followers:
            G.add_edge(follower, user)
            dfs(follower, depth + 1)
            time.sleep(1)  # DO NOT CHANGE GITHUB NEEDS A RATE LIMITER CAN CHANGE ONLY IF YOU ARE AUTHORISED

    dfs(root_user, 1)
    return G

root = "jay-sharmaa"
graph = build_follower(root, max_followers=50)

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(graph, k=1, seed=0)
nx.draw_networkx(graph, pos=pos, with_labels=True, node_color='lightgreen', arrows=True, arrowstyle='<-', node_size=1200, font_size=8)
plt.title("GitHub Follower Graph (Depth 2)")
plt.axis('off')
plt.show()