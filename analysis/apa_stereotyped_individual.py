# %%
import sys
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
import pandas as pd
from sklearn.preprocessing import StandardScaler


import numpy as np
import os
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
basepath = "/Users/federicoclaudi/Documents/Github/APA_setup"
os.chdir(basepath)

sys.path.append("./")
sys.path.append(basepath)
sys.path.append(os.path.join(basepath, "analysis"))

from fcutils.plot.distributions import plot_kde
from fcutils.plot.figure import clean_axes
from analysis.process_data import DataProcessing
from analysis.fixtures import colors, sensors
from myterial import indigo, blue_grey_dark, salmon, cyan_dark

data = DataProcessing.reload()

WT = data.data.loc[data.data.condition == "WT"].reset_index()
CTRL = data.data.loc[data.data.condition == "CTRL"].reset_index()
DTR = data.data.loc[data.data.condition == "DTR"].reset_index()
DTR_c57 = data.data.loc[(data.data.condition == "DTR")&(data.data.strain == "C57")].reset_index()
DTR_vglut = data.data.loc[(data.data.condition == "DTR")&(data.data.strain == "VGLUT")].reset_index()

data.data.groupby("condition").count()

# %%
def get_traces(condtion_trials:pd.DataFrame, t0=85, t1=140, subtractbl=True)->np.ndarray:
    """
        Get traces for all trials in a condition between the
        two reference frames. Returns an array of shape (4, n_frames, n_trials).
    """
    traces = np.zeros((4, t1 - t0, len(condtion_trials)))

    for i, trial in condtion_trials.iterrows():
        for n, paw in enumerate(sensors):
            bl = np.mean(trial[paw][0:85]) if subtractbl else 0
            traces[n, :, i] = trial[paw][t0:t1] -  bl  # remove baseline
    return traces

def get_explained_variance(model, X):
    """
        Compute the fraction of variance explained by each PC
        of `model` when transforming data `X` (a single trial/avg trace)
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

def get_trials_variance_explained(pca, traces:np.ndarray, shuffle=False)->dict:
    """
        Compute the fraction of variance explained by each PC for each trial
        in a dataset `traces`.
    """
    # get variance explained for each trial
    variance_explained = {i:[] for i in range(pca.n_components)}
    for i in range(traces.shape[-1]):
        trialtrace = traces[:,:,i].T
        if shuffle:
            trialtrace = trialtrace[:, np.random.permutation(trialtrace.shape[-1])]

        explained = get_explained_variance(pca, trialtrace)
        for i in range(pca.n_components):
            variance_explained[i].append(explained[i])
    return variance_explained


# %%
"""
PCA analysis

Fit a PCA model to the average traces of the WT trials, then compute the 
fraction of variance explained by the first PC for all WT trials and compare
to:
    - shuffled data (data in which the order of the paws is randomly shuffled)
    - control trials
    - DTR trials
"""

wt_traces_complete = get_traces(WT, t0=0, t1=239)
ctrl_traces_complete = get_traces(CTRL, t0=0, t1=239)
dtr_traces_complete = get_traces(DTR, t0=0, t1=239)
dtrc57_traces_complete = get_traces(DTR_c57, t0=0, t1=239)
dtrvglut_traces_complete = get_traces(DTR_vglut, t0=0, t1=239)

alldata = np.concatenate([wt_traces_complete, ctrl_traces_complete, dtr_traces_complete], 2).reshape(4, -1)
scaler = StandardScaler().fit(alldata.T)


# get traces between reference frames
def transform(x):
    transformed = np.zeros((x.shape[1], x.shape[0], x.shape[2]))
    for i in range(x.shape[2]):
        transformed[:, :, i] = scaler.transform(x[:, :, i].T)
    return np.transpose(transformed, (1, 0, 2))

wt_traces = transform(get_traces(WT, t0=85, t1=239))
ctrl_traces = transform(get_traces(CTRL, t0=85, t1=239))
dtr_traces = transform(get_traces(DTR, t0=85, t1=239))
dtrc57_traces = transform(get_traces(DTR_c57, t0=85, t1=239))
dtrvglut_traces = transform(get_traces(DTR_vglut, t0=85, t1=239))
allcontrols = np.dstack([wt_traces, ctrl_traces])


# get average trace for each condition
wt_avgtrace = np.mean(wt_traces, axis=2)
ctrl_avgtrace = np.mean(ctrl_traces, axis=2)
dtr_avgtrace = np.mean(dtr_traces, axis=2)
dtrc57_avgtrace = np.mean(dtrc57_traces, axis=2)
dtrvglut_avgtrace = np.mean(dtrvglut_traces, axis=2)
allcontrols_avgtrace = np.mean(allcontrols, axis=2)

# plot average traces
f, axarr = plt.subplots(2, 2, figsize=(10, 10), sharey=True, sharex=True)
axarr = axarr.flatten()
colors = (indigo, cyan_dark, salmon)

for i, ax in enumerate(axarr):
    # ax.plot(np.mean(wt_traces_complete, 2)[i, :], color=colors[0], lw=4, alpha=.15)
    # ax.plot(np.mean(ctrl_traces_complete, 2)[i, :], color=colors[1],  lw=4, alpha=.15)
    # ax.plot(np.mean(dtr_traces_complete, 2)[i, :], color=colors[2], lw=4, alpha=.15)

    ax.plot(np.arange(wt_avgtrace.shape[1]) + 85, wt_avgtrace[i, :], color=colors[0], lw=4, label="wt")
    ax.plot(np.arange(wt_avgtrace.shape[1]) + 85, ctrl_avgtrace[i, :], color=colors[1], lw=4, label="ctrl")
    # ax.plot(np.arange(wt_avgtrace.shape[1]) + 85, dtr_avgtrace[i, :], color=colors[2], lw=4, label="dtr")
    ax.plot(np.arange(wt_avgtrace.shape[1]) + 85, dtrc57_avgtrace[i, :], color="red", lw=4, label="dtr c57")
    ax.plot(np.arange(wt_avgtrace.shape[1]) + 85, dtrvglut_avgtrace[i, :], color="green", lw=4, label="dtr vglut")

axarr[0].legend()





# %%
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def plot_variance_kde(ax, variance_explained:dict, color="k", label="", alpha=0.1, n_pc=2):
    """
        Plot the variance explained by each PC as a KDE
    """
    variance_explained = np.vstack(list(variance_explained.values())).T  # turn into a numpy array
    X = np.sum(variance_explained[:, 0:n_pc], 1) # sum first n PCs
    # X = variance_explained[:, 1]
    plot_kde(ax, data=X, color=color, label=label, lw=3, alpha=alpha, kde_kwargs=dict(bw=.05))
    ax.plot([np.median(X), np.median(X)], [0, 3.5], color=color, lw=4, solid_capstyle="round")    



# compare PCA to shuffled data
pca = PCA(n_components=4).fit(allcontrols_avgtrace.T)

# compute variance explained for mean trace
print(f'Explained variation per principal component on average wt_traces: {pca.explained_variance_ratio_}')

# get variance explained for each trial
variance_explained = get_trials_variance_explained(pca, allcontrols)
variance_explained_shuffled = get_trials_variance_explained(pca, allcontrols, shuffle=True)

# PLot fraction of variance explained by first PC on WT and shuffled trials
f, ax = plt.subplots(figsize=(16, 9))
plot_variance_kde(ax, variance_explained_shuffled, color=blue_grey_dark, label="PC 1 - shuffked", alpha=.05)
plot_variance_kde(ax, variance_explained, color=indigo, label="PC 1 - trials", )

_, pval = ttest(variance_explained[0], variance_explained_shuffled[0])
ax.set(xlim=[0, 1], xlabel="fraction of varince explained", ylabel="density", title=f"Pvalue: {pval:.4e} (significant: {pval < 0.05})")
clean_axes(f)
_ = ax.legend()

# %%
# get variance explained for each trial
variance_explained_ctrl = get_trials_variance_explained(pca, ctrl_traces)
variance_explained_dtr = get_trials_variance_explained(pca, dtr_traces)
variance_explained_dtrc57 = get_trials_variance_explained(pca, dtrc57_traces)
variance_explained_dtrvglut = get_trials_variance_explained(pca, dtrvglut_traces)
variance_explained_allctrl = get_trials_variance_explained(pca, allcontrols)

# PLot fraction of variance explained by first PC on WT and shuffled trials
f, ax = plt.subplots(figsize=(16, 9))
# plot_variance_kde(ax, variance_explained_dtr, color=salmon, label="PC 1 - dtr trials", )
# plot_variance_kde(ax, variance_explained_ctrl, color=cyan_dark, label="PC 1 - ctrl trials", )
# plot_variance_kde(ax, variance_explained, color=indigo, label="PC 1 - trials", )
plot_variance_kde(ax, variance_explained_allctrl, color="black", label="PC 1 - allcontrols", )
plot_variance_kde(ax, variance_explained_dtrc57, color="red", label="PC 1 - dtr c57", )
plot_variance_kde(ax, variance_explained_dtrvglut, color="green", label="PC 1 - dtr vglut", )

# ax.plot([pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[0]], [0, 3], color=salmon, solid_capstyle="round", lw=6, label=f"PC 1 - mean")

_, pval_dtr = ttest(variance_explained[0], variance_explained_dtr[0])
_, pval_ctrl = ttest(variance_explained[0], variance_explained_ctrl[0])
_, pval_dtrctrl = ttest(variance_explained_dtr[0], variance_explained_ctrl[0])

title = f"""
Pvalue (wt-ctrl): {pval_ctrl:.4e} (significant: {pval_ctrl < 0.05})
Pvalue (wt-dtr): {pval_dtr:.4e} (significant: {pval_dtr < 0.05})
Pvalue (dtr-ctrl): {pval_dtrctrl:.4e} (significant: {pval_dtrctrl < 0.05})
"""
ax.set(xlim=[0, 1], xlabel="fraction of varince explained", ylabel="density", title=title)
clean_axes(f)
ax.legend()



# %%
# group average and trials traces
# averages = (allcontrols_avgtrace, dtr_avgtrace)
# trials = (allcontrols, dtr_traces)
# names = ("CTRL", "DTR")
# colors = ("m", salmon)

# f, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
# components = pca.components_

# for n, (color, X, x) in enumerate(zip(colors, averages, trials)):
#     for i in range(x.shape[2]):
#         pc = pca.transform(x[:, :, i].T)

#         axes[0, n].plot(pc[:, 0], color="k", lw=.5)
#         axes[1, n].plot(pc[:, 1], color="k", lw=.5)

#     axes[0, n].plot(pca.transform(X.T)[:, 0], lw=6, color=color)
#     axes[1, n].plot(pca.transform(X.T)[:, 1], lw=6, color=color)





# %%
"""
Plot data in PCA space
"""
# f, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True, sharey=True)
# axes = axes.flatten()

f, ax = plt.subplots(figsize=(10, 10))


averages = (dtrc57_avgtrace, dtrvglut_avgtrace, ctrl_avgtrace,)
names = ("DTR c57", "DTR vglut", "CTRL", "DTR")
colors = ("red", "green", "black")

for n, (name, color, X) in enumerate(zip(names, colors, averages)):
    # plot each trial in PC space
    # for i in range(x.shape[2]):
    #     # project each trial into PC space
    #     trial_pca = pca.transform(x[:, :, i].T)
    #     # plot each trial in PC space
    #     axes[n].plot(trial_pca[:, 0], trial_pca[:, 1], color=color, alpha=.4, lw=1.5)

    # plot each basis vector
    if n == 0:
        vecs = np.vstack([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
        vecs_pcs = pca.transform(vecs)
        for i in range(4):
            ax.plot([0, vecs_pcs[0, i]], [0, vecs_pcs[1, i]], label=sensors[i], lw=4, alpha=.3)
        
    # plot average trace
    pc = pca.transform(X.T)
    
    ax.plot(pc[:, 0], pc[:, 1], lw=6, color=color, label=name)

    ax.legend()
    ax.set(xlabel="PC1", ylabel="PC2", xlim=[-2, 3], ylim=[-2, 3])



# %%
# ---------------------------------------------------------------------------- #
#                             CORRELATION ANALYSIS                             #
# ---------------------------------------------------------------------------- #

# wt_traces = get_traces(WT, t0=85, t1=239, subtractbl=False)
# ctrl_traces = get_traces(CTRL, t0=85, t1=239, subtractbl=False)
# dtr_traces = get_traces(DTR, t0=85, t1=239, subtractbl=False)


# function to compute pearson correlation between two time series
def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

f, ax = plt.subplots(figsize=(16, 9))

# compute pearson correlation for each trial in each condition
allcontrols = np.dstack([wt_traces, ctrl_traces])

paw1 = 0
paw2 = 2

wt_corrs = [pearson_corr(wt_traces[paw1, :, i], wt_traces[paw2, :, i]) for i in range(wt_traces.shape[2])]
ctrl_corrs = [pearson_corr(ctrl_traces[paw1, :, i], ctrl_traces[paw2, :, i]) for i in range(ctrl_traces.shape[2])]
dtr_corrs = [pearson_corr(dtr_traces[paw1, :, i], dtr_traces[paw2, :, i]) for i in range(dtr_traces.shape[2])]
dtr_c57_corrs = [pearson_corr(dtrc57_traces[paw1, :, i], dtrc57_traces[paw2, :, i]) for i in range(dtrc57_traces.shape[2])]
dtr_vglut_corrs = [pearson_corr(dtrvglut_traces[paw1, :, i], dtrvglut_traces[paw2, :, i]) for i in range(dtrvglut_traces.shape[2])]
allcontrols_corrs = [pearson_corr(allcontrols[paw1, :, i], allcontrols[paw2, :, i]) for i in range(allcontrols.shape[2])]

# _, pval_dtr = ttest(wt_corrs, dtr_corrs)
# _, pval_ctrl = ttest(wt_corrs, ctrl_corrs)
# _, pval_dtrctrl = ttest(dtr_corrs, ctrl_corrs)

_, pval_c57 = ttest(dtr_c57_corrs, allcontrols_corrs)
_, pval_vglut = ttest(dtr_vglut_corrs, allcontrols_corrs)

title = f"""
Pvalue (dtr c57-all ctrl): {pval_c57:.4e} (significant: {pval_c57 < 0.05})
Pvalue (dtr vglut-all ctrl): {pval_vglut:.4e} (significant: {pval_vglut < 0.05})
"""

names = ("DTR c57", "DTR vglut",  "ALLCTRL")
colors = ("red", "green",  "black")
correlations = (dtr_c57_corrs, dtr_vglut_corrs,  allcontrols_corrs)

for n, (name, color, corrs) in enumerate(zip(names, colors, correlations)):
    plot_kde(ax, data=corrs, color=color, label=name, lw=3, alpha=.3, kde_kwargs=dict(bw=.05))

ax.set(xlim=[-1, 1], ylim=[0, 5], xlabel="Pearson correlation", ylabel="Density", title=title)
ax.legend()

# %%
# look at correlations between all paws at the same time


allctrls_corrs = np.vstack([np.corrcoef(allcontrols[:, :, i])[0, :] for i in range(allcontrols.shape[2])])
dtr_corrs = np.vstack([np.corrcoef(dtr_traces[:, :, i])[0, :] for i in range(dtr_traces.shape[2])])

f, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, sharey=True)

axes[0].imshow(allctrls_corrs.T, vmin=-1, vmax=1, cmap="bwr")
axes[1].imshow(dtr_corrs.T, vmin=-1, vmax=1, cmap="bwr")




# %%
# %%
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
