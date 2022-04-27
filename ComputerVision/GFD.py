import numpy as np
from numpy import fft
from PIL import Image
from matplotlib import pyplot as plt
import math as math


def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def GFD(bw, m, n):
    # preprocess the image
    if len(bw.shape) > 2:
        bw = bw.max(axis = 2) / 255
    width = bw.shape[1]
    N = width
    maxR = math.sqrt((((N)//2)**2) + (((N)//2)**2))

    x = np.linspace(-(N-1)//2, (N-1)//2, N )
    y = x
    X, Y = np.meshgrid(x, y)

    radius = np.sqrt(np.power(X, 2) + np.power(Y, 2)) / maxR

    theta = np.arctan2(Y, X)
    theta[theta < 0] = theta[theta < 0] + 2+ np.pi

    FR = np.zeros((m,n))
    FI = np.zeros((m,n))
    FD = np.zeros((m*n,1))

    i = 0
    for rad in range(m):
        for ang in range(n):
            # e^(i * theta) = cos(theta) + i * sin(theta)
            # PF = FR + i * FI

            tempR = bw * np.cos(2 * np.pi * rad * radius + ang * theta)
            tempI = bw * np.sin(2 * np.pi * rad * radius + ang * theta)
            FR[rad, ang] = np.sum(tempR)
            FI[rad, ang] = np.sum(tempI)
            
            if rad == 0 and ang == 0:
                FD[i] = math.sqrt((2* (FR[0,0] * FR[0,0]))) / (np.pi* maxR * maxR)
            else:
                FD[i] = math.sqrt((FR[rad, ang] * FR[rad, ang]) + (FI[rad, ang] * FI[rad, ang])) / (math.sqrt((2* (FR[0,0] * FR[0,0]))))
            i = i + 1
    plt.figure()
    plt.imshow(FD)
    plt.show()
    return FD

image_names = [
    'facebook',
    'instagram1',
    'instagram2',
    'tennis1',
    'tennis3',
    'twitter1'
]

images = [{ 'path':  'images/' + image + '.png', 'image': np.array(Image.open('./images/' + image + '.png')  ) } for image in image_names ]

db = [ { 'path': image['path'], 'image': image['image'], 'descriptor': GFD(image['image'], 4, 9)} for image in images]


def search(image_name):
    stats = []
    image = np.array(Image.open('images/'+image_name+'.png'))
    gfd = GFD(image, 4, 9)
    for db_image in db:
        stats.append({'distance': manhattan_distance(gfd, db_image['descriptor']), 'path': db_image['path'] })
        stats.sort(key=distance)
    return stats

def distance(stat):
    return stat['distance']

if __name__=="__main__":
    img_path="images/facebook.png"
    stats=search("facebook")
    print(stats)
    print(distance(stats[0]))