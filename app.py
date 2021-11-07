from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('rFPki.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color


modified_img = cv2.resize(image, (900, 600), interpolation = cv2.INTER_AREA)
modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)


clf = KMeans(n_clusters=5)
color_labels = clf.fit_predict(modified_img)
center_colors = clf.cluster_centers_
counts = Counter(color_labels)
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
plt.figure(figsize=(12, 8))
plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
plt.savefig("color_analysis_report.png")
print(hex_colors)
