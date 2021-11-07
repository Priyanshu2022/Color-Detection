# Dataset - https://github.com/codebrainz/color-names/blob/master/output/colors.csv

import cv2
import pandas as pd
from colorthief import ColorThief

from collections import Counter
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


image = cv2.imread('rFPki.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color


# modified_img = cv2.resize(image, (900, 600), interpolation = cv2.INTER_AREA)
# modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
# clf = KMeans(n_clusters=5)
# color_labels = clf.fit_predict(modified_img)
# center_colors = clf.cluster_centers_
# counts = Counter(color_labels)
# ordered_colors = [center_colors[i] for i in counts.keys()]
# hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]

#
# plt.figure(figsize=(12, 8))
# plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
# plt.savefig("color_analysis_report.png")
# print(hex_colors)



# img_path = r'C:\Users\Balaji\Downloads\color detection\colorpic'
img = cv2.imread('rFPki.jpg')
color_thief = ColorThief('rFPki.jpg')
# declaring global variables (are used later on)
clicked = False
r = g = b = x_pos = y_pos = 0

# Reading csv file with pandas and giving names to each column
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('colors.csv', names=index, header=None)


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# def rgb_to_hex(r, g, b):
#   return ('{:X}{:X}{:X}').format(r, g, b)


#getting palette of of top 5 dominant color in rgb format
palette = color_thief.get_palette(color_count=5)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)


while True:

    cv2.imshow("image", img)
    if clicked:

        # cv2.rectangle(image, start point, endpoint, color, -1 fills entire rectangle)
        cv2.rectangle(img, (20, 20), (990, 60), (b, g, r), -1)


        cv2.circle(img, (30,600), 25, (palette[0][0], palette[0][1], palette[0][2]), -1)
        cv2.circle(img, (80,600), 25, (palette[1][0], palette[1][1], palette[1][2]), -1)
        cv2.circle(img, (130,600), 25, (palette[2][0], palette[2][1], palette[2][2]), -1)
        cv2.circle(img, (180,600), 25, (palette[3][0], palette[3][1], palette[3][2]), -1)
        cv2.circle(img, (230,600), 25, (palette[4][0], palette[4][1], palette[4][2]), -1)
        cv2.putText(img,'Color Palette',(10,550),2,0.8,(255,255,255),2,cv2.LINE_AA)

        # Creating text string to display( Color name and RGB values )
        text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b) + ' Color Code = '+rgb_to_hex(r,g,b);

        # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text, (50, 50), 2, 0.8, (255,255,255), 2, cv2.LINE_AA)


        # For very light colours we will display text in black colour
        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    # Break the loop when user hits 'esc' key
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
