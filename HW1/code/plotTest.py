# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:00:56 2018

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.plot(t, s)
ax.plot(t, 2 + np.sin(t) )
ax.plot ([1,2])
ax.abline (intercept=0, slope=1)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='2 Times Pi Times time')
ax.grid()

#fig.savefig("test.png")
plt.show()