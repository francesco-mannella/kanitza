import json
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
filePathName = "controllable.json"

with open(filePathName, "r") as json_file:
    jsw = json.load(json_file)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, aspect="equal")
ax.set_xlim([-30, 200])
ax.set_ylim([-30, 200])
for obj in jsw["body"]:
    print(obj["name"])
    ax.set_title(obj["name"])
    x = np.array(obj["fixture"][0]["polygon"]["vertices"]["x"])
    x += obj["position"]["x"]
    y = np.array(obj["fixture"][0]["polygon"]["vertices"]["y"])
    y += obj["position"]["y"]
    
    print(x)
    ax.plot(x, y)
    input()
