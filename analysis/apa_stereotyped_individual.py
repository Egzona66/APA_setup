# %%
import sys
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
from sklearn.linear_model import LinearRegression

import numpy as np
import os

basepath = "/Users/federicoclaudi/Documents/Github/APA_setup"
os.chdir(basepath)

sys.path.append("./")
sys.path.append(basepath)
sys.path.append(os.path.join(basepath, "analysis"))

from fcutils.plot.distributions import plot_kde
from fcutils.plot.figure import clean_axes
from analysis.process_data import DataProcessing
from analysis.fixtures import colors, sensors
from myterial import indigo, blue_grey_dark, salmon

data = DataProcessing.reload()


# %%
# get start frame for decrease of FR

FR = np.vstack([t.hl for (i,t) in data.data.iterrows()]).T
mu = np.mean(FR, 1)

plt.plot(FR, color="k")
plt.plot(mu, color="red", lw=3)

plt.plot(np.diff(mu), color="green", lw=5)
plt.axvline(85,)
# print(f"First descent: {np.where(np.diff(mu) < -.5)[0][0]+1}")

# %%
# extract slopes
f, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 12), sharey=True, sharex=True)
axes = axes.flatten()

f2, ax2 = plt.subplots(figsize=(14, 9))

# def slope(y1, y2, t0, t1):
#     return (y2-y1)/(t1-t0)

def slope(t0, t1, Y):
    """
        Get the slope of a linear regression
    """
    X = np.arange(t0, t1)
    # print(X.shape)
    # print(Y.shape)
    return LinearRegression().fit(X.reshape(-1, 1), Y.reshape(-1, 1)).coef_[0][0]

criteria = dict(
    fl=[10, 20],
    fr=[10, 20],
    hl=[20, 40],
    hr=[20, 40],
)

blt0, blt1 = 0, 85
t0, t1 = 85, 140
slopes, shuffledslopes, baselineslopes = {p:[] for p in sensors}, {p:[] for p in sensors}, {p:[] for p in sensors}
for i, trial in data.data.iterrows():
    
    use = True
    # for paw, limits in criteria.items():
    #     baseline = np.mean(trial[paw][blt0:blt1])
    #     if baseline > limits[1] or baseline < limits[0]:
    #         use = False
    # if not use:
    #     continue

    for n, paw in enumerate(sensors):         

        # baseline = np.mean(trial[paw][blt0:blt1])
        # baseline = trial[paw][0]
        baseline = 0
        trace = trial[paw] - baseline

        axes[n].plot(trace, lw=3, color=colors[paw])
        axes[n].plot([t0, t1], [trace[t0], trace[t1]], color="red", zorder=100, alpha=.25, lw=4)
        axes[n].plot([blt0, blt1], [trace[blt0], trace[blt1]], color="k", zorder=100, alpha=.25, lw=4)

        # get slopes
        # slopes[paw].append(slope(trial[paw][t0], trial[paw][t1], t0, t1))
        # baselineslopes[paw].append(slope(trial[paw][blt0], trial[paw][blt1], blt0, blt1))

        # shuffled = np.random.permutation(trial[paw])
        # shuffledslopes[paw].append(slope(shuffled[t0], shuffled[t1], t0, t1))

        slopes[paw].append(slope(t0, t1,trace[t0:t1]))
        baselineslopes[paw].append(slope(blt0, blt1, trace[blt0:blt1]))

        shuffled = np.random.permutation(trace)
        shuffledslopes[paw].append(slope(t0, t1,shuffled[t0:t1]))

        # make pretty plots
        ax2.plot(
            [.25+n, .75+n], [trace[t0], trace[t1]], "o-", lw=2, zorder=100, color="red"
        )
        ax2.plot(
            [.25+n, .75+n], [trace[blt0], trace[blt1]], "o-", lw=2, zorder=-1, color=[.55, .55, .55],
        )

    for ax in axes:
        ax.axvline(120, lw=2, ls="--", color="k")

_ = ax2.set(xticks=[.5, 1.5, 2.5, 3.5], xticklabels=sensors)


# %%
# inspect slopes distributions
from statsmodels.sandbox.stats.multicomp import multipletests

f, axes = plt.subplots(ncols=2, nrows=2, figsize=(12, 14), sharey=True, sharex=True)
axes = axes.flatten()

bins = np.linspace(-1, 1, 25)
pvals = []
for n, paw in enumerate(sensors):   
    for i, (_slopes, color, name) in enumerate(zip(
                            [slopes, baselineslopes], 
                            ("red", "blue"),
                            ["slopes", "baselineslopes"], 
                            )):
        axes[n].hist(_slopes[paw], bins=bins, color=color, alpha=.5, density=True, zorder=-1)
        plot_kde(axes[n], data=_slopes[paw], kde_kwargs=dict(bw=.075), label=name, color=color, lw=3)

        axes[n].plot(
            [np.mean(_slopes[paw])-.025, np.mean(_slopes[paw])+0.025], 
            [-.25 * (i+1), -.25 * (i+1)], lw=4, alpha=1, zorder=100, color="white", solid_capstyle="round")
        axes[n].plot(
            [np.percentile(_slopes[paw], 5), np.percentile(_slopes[paw], 95)],
            [-.25 * (i+1), -.25 * (i+1)],  lw=6, alpha=.8, color=color, solid_capstyle="round")


    axes[n].axhline(0, lw=1, color="black")
    axes[n].legend()
    
    
    stat, p = ttest(slopes[paw], baselineslopes[paw])
    pvals.append(p)

sig, corr, _, _ = multipletests(pvals)
print(f"Corrected pvalues: {corr}\n, significant {sig}")

for n, ax in enumerate(axes):
    ax.set(xlim=[-1.0, 1.0], ylim=[-1, 8], title=f"Paw: {sensors[n]}, pval: {corr[n]:.4e} - significant: {sig[n]}")


# %%
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d

"""
PCA analysis

We fit PCA to the average trace for the 4 paws in the frames ranges we care about
(N x 4 array) and we look at the fraction of variance explained by each PC
and plot the stuff in the first 3 PC.

"""

def get_explained_variance(model, X):
    """
        Compute the fraction of variance explained by each PC
        of `model` when transforming data `X
    """
    result = np.zeros(model.n_components)
    for ii in range(model.n_components):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)

        result[ii] = 1 - (np.linalg.norm(X_approx_ii - X) /
                          np.linalg.norm(X - model.mean_)) ** 2
    return result


# f, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
# axes=axes.flatten()
t0, t1 = 85, 170

traces = np.zeros((4, t1 - t0, len(data.data)))

for i, trial in data.data.iterrows():
    for n, paw in enumerate(sensors):
        traces[n, :, i] = trial[paw][t0:t1]
        # axes[n].plot(trial[paw], color="k", zorder=-1)
        # axes[n].plot(np.arange(t0, t1),trial[paw][t0:t1], color="red", zorder=100)


f, axes = plt.subplots(figsize=(12, 12), nrows=2)


avg = np.mean(traces, axis=2)

pca = PCA(n_components=4).fit(avg.T)

variance_explained = {i:[] for i in range(pca.n_components)}
variance_explained_shuffled = {i:[] for i in range(pca.n_components)}

for i in range(traces.shape[-1]):
    trialtrace = traces[:,:,i].T
    trialshuffled = trialtrace[:, np.random.permutation(trialtrace.shape[-1])]

    pc = pca.transform(trialtrace)
    explained = get_explained_variance(pca, trialtrace)
    explained_shuffled = get_explained_variance(pca, trialshuffled)

    for i in range(pca.n_components):
        variance_explained[i].append(explained[i])
        variance_explained_shuffled[i].append(explained_shuffled[i])
        
        
    axes[0].plot(pc[:, 0])
    axes[1].plot(pc[:, 1])

# compute variance explained
print('Explained variation per principal component on average traces: {}'.format(pca.explained_variance_ratio_))


# %% 
# PLot fraction of variance explained by each PC
f, ax = plt.subplots(figsize=(16, 9))


# ax.hist(variance_explained[0], bins=np.linspace(0, 1, 10), label=f"PC 1 - trials", alpha=.4)
plot_kde(ax, data=variance_explained[0], kde_kwargs=dict(bw=.06), color=indigo, label=f"PC 1 - trials", lw=3, alpha=.4)
plot_kde(ax, data=variance_explained_shuffled[0], kde_kwargs=dict(bw=.06), color=blue_grey_dark, label=f"PC 1 - shuffled", lw=3)

ax.plot([pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[0]], [0, 3], color=salmon, solid_capstyle="round", lw=6, label=f"PC 1 - mean")

for n, (x, color) in enumerate(zip((variance_explained, variance_explained_shuffled), (indigo, blue_grey_dark))):
    y = -.1 * (n+1)
    ax.plot([np.median(x[0]), np.median(x[0])], [0, 2.5], color=color, lw=4, solid_capstyle="round")    

_, pval = ttest(variance_explained[0], variance_explained_shuffled[0])

ax.set(xlim=[0, 1], xlabel="fraction of varince explained", ylabel="density", title=f"Pvalue: {pval:.4e} (significant: {pval < 0.05})")
clean_axes(f)
ax.legend()


# %%
""" plot PCA in 3D"""
fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection='3d')
ax.view_init(45, 70)
ax.set(xlim=[-80, 20], ylim=[-80, 80], zlim=[-80, 80])



for i in range(traces.shape[-1]):
    pc = pca.transform(traces[:,:,i].T)
    ax.scatter(pc[:, 0] - pc[0, 0], pc[:, 1] - pc[0, 1], pc[:, 2] - pc[0, 2], color="k")

ax.plot([0, -60], [0, 0], [0, 0], lw=8, zorder=30, solid_capstyle="round", color="red")
ax.plot([0, 0], [0, 20], [0, 0], lw=8, zorder=40, solid_capstyle="round", color="blue")
ax.plot([0, 0], [0, 0], [0, 20], lw=8, zorder=40, solid_capstyle="round", color="green")



# # %%
# """
# Clustering analysis using SOM

# https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
# https://github.com/JustGlowing/minisom
# """
# import math
# from minisom import MiniSom
# from tslearn.barycenters import dtw_barycenter_averaging
# from tslearn.clustering import TimeSeriesKMeans
# from sklearn.cluster import KMeans

# mySeries = [trial.hl for i, trial in data.data.iterrows()]

# # som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
# som_x, som_y = 2,   2

# som = MiniSom(2, 2, len(mySeries[0]), sigma=0.1, learning_rate = 0.1)

# som.random_weights_init(mySeries)
# som.train(mySeries, 50000)



# # %%
# def plot_som_series_averaged_center(som_x, som_y, win_map):
#     fig, axs = plt.subplots(som_x,som_y, figsize=(25,25))
#     fig.suptitle('Clusters')
#     for x in range(som_x):
#         for y in range(som_y):
#             cluster = (x,y)
#             if cluster in win_map.keys():
#                 for series in win_map[cluster]:
#                     axs[cluster].plot(series,c="gray",alpha=0.5) 
#                 axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
#             cluster_number = x*som_y+y+1
#             axs[cluster].set_title(f"Cluster {cluster_number}")

#     plt.show()

# def plot_som_series_dba_center(som_x, som_y, win_map):
#     fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
#     fig.suptitle('Clusters')
#     for x in range(som_x):
#         for y in range(som_y):
#             cluster = (x,y)
#             if cluster in win_map.keys():
#                 for series in win_map[cluster]:
#                     axs[cluster].plot(series,c="gray",alpha=0.5) 
#                 axs[cluster].plot(dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red") # I changed this part
#             cluster_number = x*som_y+y+1
#             axs[cluster].set_title(f"Cluster {cluster_number}")

#     plt.show()

# win_map = som.win_map(mySeries)
# # Returns the mapping of the winner nodes and inputs

# plot_som_series_averaged_center(som_x, som_y, win_map)
# # %%

# """
# K-means clustering
# """

# # cluster_count = math.ceil(math.sqrt(len(mySeries))) 
# cluster_count = 3
# # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

# km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

# labels = km.fit_predict(mySeries)

# plot_count = math.ceil(math.sqrt(cluster_count))

# fig, axs = plt.subplots(3, 2,figsize=(25,25))
# fig.suptitle('Clusters')
# row_i=0
# column_j=0
# # For each label there is,
# # plots every series with that label
# for label in set(labels):
#     cluster = []
#     for i in range(len(labels)):
#             if(labels[i]==label):
#                 axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
#                 cluster.append(mySeries[i])
#     if len(cluster) > 0:
#         axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
#     axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
#     column_j+=1
#     if column_j%plot_count == 0:
#         row_i+=1
#         column_j=0
        
# # %%
