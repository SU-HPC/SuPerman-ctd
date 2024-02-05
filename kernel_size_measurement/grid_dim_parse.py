import os
import sys
import glob
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

def parse_this_file(parsed, fname):

    grid_dim = int(fname.split("_")[1].strip(".txt"))
    
    reader = open(fname, "r")
    lines = reader.readlines()

    first = True
    
    this_results = []
    for line in lines:
        if(line.find("Result") != -1):
            if(not first):
                time = float(line.split(" in ")[1])
                this_results.append(time)
            else:
                first = False

    parsed.append([grid_dim, mean(this_results)])
    return parsed

def parse_this_size(size):

    regex = "*" + "CYCLE" + str(size) + "*"

    files = []
    for fname in glob.glob(regex):
        files.append(fname)

    print("regex:", regex)
    print("files:", files)
    

    parsed = []
    for fname in files:
        parsed = parse_this_file(parsed, fname)

    parsed = sorted(parsed, key=lambda x: x[0])
    for item in parsed:
        print(item)


    return parsed


def annot_min(x,y,bc,ax=None):
    xmin = x[np.argmin(y)]
    ymin = y.min()
    #text= "x={:}, y={:.3f}".format(xmin, ymin)
    text= "gs={:}, time={:.3f}, s_bc={:.2E}".format(xmin, ymin, bc)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="r", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=95")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.94,0.96), **kw)

def annot_min_c(xmin, y, bc, ax=None):
    ymin = y[xmin - 40]
    #print('xminc:', xmin, 'yminc:', ymin)
    text= "gs={:}, time={:.3f}, s_bc={:.2E}".format(xmin, ymin, bc)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="b", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=95")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.1,1.2), **kw)

def annot_min_j(xmin, y, bc, ax):
    ymin = y[xmin - 40]
    #print('xminc:', xmin, 'yminc:', ymin)
    #text= "x={:}, y={:.3f}".format(xmin, ymin)
    text= "gs={:}, time={:.3f}, s_bc={:.2E}".format(xmin, ymin, bc)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="g", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=95")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.4,1.2), **kw)

    
def get_size_vals(size):

    this_size_parsed = parse_this_size(size)

    x = []
    y = []

    for item in this_size_parsed:
        x.append(item[0])
        y.append(item[1])


    nx = np.array(x)
    ny = np.array(y)

    return nx, ny


def draw_one_size(ax, nx, ny, c, j, bcm, bcc, bcj, size):

    ax.plot(nx, ny)
    ax.set_ylabel(str(size))
    annot_min(nx,ny, bcm, ax)
    annot_min_c(c, ny, bcc, ax)
    annot_min_j(j, ny, bcj, ax)
    
        
def main():

       
    fig, axs = plt.subplots(6, sharex=True)

    # Set common labels
    fig.text(0.5, 0.04, 'Grid Dimension', ha='center', va='center')
    fig.text(0.06, 0.5, 'Time / Matrix Size', ha='center', va='center', rotation='vertical')

    plt.xticks([40, 80, 85, 160, 240, 320, 350], ['40', '80', '85', '160', '240', '320', '350'])

    nx, ny = get_size_vals(30)
    draw_one_size(axs[0], nx, ny, 80, 85, 233357, 231898, 232000, 30)

    nx, ny = get_size_vals(31)
    draw_one_size(axs[1], nx, ny, 80, 85, 241869, 477701, 477848, 31)

    nx, ny = get_size_vals(32)
    draw_one_size(axs[2], nx, ny, 80, 85, 986198, 492615, 492622, 32)

    nx, ny = get_size_vals(33)
    draw_one_size(axs[3], nx, ny, 80, 85, 1015783, 2029170, 2029297, 33)

    nx, ny = get_size_vals(34)
    draw_one_size(axs[4], nx, ny, 80, 85, 4179677, 4179677, 4179821, 34)

    nx, ny = get_size_vals(35)
    draw_one_size(axs[5], nx, ny, 80, 85, 4182990, 4179695, 4179749, 35)

    plt.show()


if __name__ == "__main__":

    main()
