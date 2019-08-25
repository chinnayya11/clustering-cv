# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:16:32 2019

@author: shloka
"""

from sklearn import cluster , datasets
import pandas as pd
import numpy as np
#import pdb; pdb.set_trace()
import matplotlib.pyplot as plt
x=pd.read_csv("chennai.csv")

def dist(xa,ya,xb,yb):
        dist=np.sqrt(((xa-xb)*(xa-xb))+((ya-yb)*(ya-yb)))
        return dist

ram,prc=[],[]
for i in x.itertuples():
    # print(i)
    # import pdb; pdb.set_trace()
    prc.append(i[21])
    ram.append(i[14])
    
plt.scatter(prc,ram)
plt.show()
k_means = cluster.KMeans(n_clusters=3)
# k_means.fit(X)
data=pd.DataFrame({
    'x':ram,
    'y':prc
})
#import pdb; pdb.set_trace()
print(k_means.fit(data))
labels = k_means.predict(data)
centroids = k_means.cluster_centers_
fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = map(lambda x: colmap[x+1], labels)
for i in data.itertuples():
    if dist(i[1],i[2],centroids[0][0],centroids[0][1])== min(dist(i[1],i[2],centroids[0][0],centroids[0][1]),dist(i[1],i[2],centroids[1][0],centroids[1][1]),dist(i[1],i[2],centroids[2][0],centroids[2][1])):
        plt.scatter(i[1],i[2],color='r')
    elif dist(i[1],i[2],centroids[1][0],centroids[1][1])== min(dist(i[1],i[2],centroids[0][0],centroids[0][1]),dist(i[1],i[2],centroids[1][0],centroids[1][1]),dist(i[1],i[2],centroids[2][0],centroids[2][1])):
        plt.scatter(i[1],i[2],color='b')
    else:
        plt.scatter(i[1],i[2],color='g')
        
plt.show()