# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:25:06 2018

@author: Marcus
"""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(2, 100)
print (data)
#data = ([0.1, 0.2, 0.3],[0.1,0.2,0,3])
#data = (0.3,0.1)


fig, axs = plt.subplots(2, 2, figsize=(5, 5))
axs[0, 0].hist(data[1])
axs[1, 0].scatter(data[1], data[0], marker='*', c='blue')
#axs[1, 0].hist2d(data[0], data[1], bins=10)
axs[0, 1].plot(data[1], data[0])
axs[1, 1].hist2d(data[0], data[1])
#axs.hist2d(data[0], data[1])
#plt.plot (0, 0, 100)

plt.show()