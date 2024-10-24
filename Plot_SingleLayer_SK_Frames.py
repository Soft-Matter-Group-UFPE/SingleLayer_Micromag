#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:52:33 2023
@author: matheus
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt

parentdir = os.getcwd()

# accessing the mz folder
os.chdir("./mz/") # you can change here for mx and my, if you want

# searching for the magnetization files and ordering them in ascending order
mz = [filename for filename in os.listdir('.') if filename.startswith("mz_")]
mz=sorted(mz, key=lambda s: int(re.search(r'\d+', s).group()))

print("Ploting...")

# looping through all the files in ascending time order
for i in range(0,len(mz)):
    os.chdir(parentdir)
    os.chdir("./mz/")

    # import file content as a matrix
    plot_mz = np.loadtxt(mz[i])

    # create a figure with a subplot
    fig = plt.figure(figsize = (10,10)) 
    ax = fig.add_subplot(2,1,1)
    
    # take the matrix and generate a heatmap with -1 and 1 as limits
    heatmap = ax.imshow(plot_mz,vmin=-1.0,vmax=1.0, cmap = 'coolwarm' , interpolation = 'nearest' )

    # adding a color bar with range from -1 to 1
    plt.colorbar(heatmap,ax=ax,ticks=[-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.50,0.75,1.00])
    
    # we need to invert the y-axis order because python is reading in a opposite direction
    ax.invert_yaxis()

    # limit the size of the simulation box to match the size of magnetization matrices
    ax.set_xlim(xmax=len(plot_mz)+1); ax.set_ylim(ymax=len(plot_mz)+1)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        
    # save our Figure at time i
    os.chdir(parentdir)
    os.chdir("./Figures_Mag")
    plt.savefig("./Plot_%d" %i + ".png",dpi=200,bbox_inches='tight')
    plt.close()
