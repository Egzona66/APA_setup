import matplotlib as mpl
import sys

if sys.platform == "darwin":
    mpl.use("Qt5Agg")


font = {'family' : 'Courier New',
        'weight' : 600,
        'size'   : 22}

mpl.rc('font', **font)


# Set up matplotlib
mpl.rcParams['text.color'] = [1, .5, 0]

mpl.rcParams['figure.figsize'] = [24, 18]
mpl.rcParams['figure.facecolor'] = [.12, .12, .12]
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['figure.dpi'] = 80
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.facecolor'] = [.12, .12, .12]
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['figure.titleweight'] = 'bold'

mpl.rcParams['lines.linewidth'] = 2.0

mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.numpoints'] = 2
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['legend.framealpha'] = .8
mpl.rcParams['legend.scatterpoints'] = 3
mpl.rcParams['legend.edgecolor'] = [.7, .7, .7]
mpl.rcParams['legend.facecolor'] = [.25, .25, .25]
mpl.rcParams['legend.shadow'] = True
mpl.rcParams['legend.columnspacing'] = 1

mpl.rcParams['axes.facecolor'] = [.2, .2, .2]
mpl.rcParams['axes.edgecolor'] = [.8, .8, .8]
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.labelcolor'] = [1, .5, 0]
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.labelweight'] = "bold"
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.titleweight'] = 800
mpl.rcParams['axes.titlepad'] = 12.

mpl.rcParams['xtick.color'] = [.8, .8, .8]
mpl.rcParams['xtick.major.size'] = 16
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['xtick.direction'] = "inout"
mpl.rcParams['xtick.labelsize'] = 8

mpl.rcParams['ytick.color'] = [.8, .8, .8]
mpl.rcParams['ytick.major.size'] = 16
mpl.rcParams['ytick.major.width'] = 3
mpl.rcParams['ytick.direction'] = "inout"
mpl.rcParams['ytick.labelsize'] = 8



import matplotlib.pyplot as plt


