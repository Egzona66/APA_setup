# %%
f, ax = plt.subplots()
diffs = [hl - fr for hl, fr in zip(hls, frs)]
for fr, hl in zip(frs, hls):
    ax.plot(hl-fr)

#%%
# Get just the CG for the frames of interest
cgs = [r.cg[r.start:r.end, :]-r.cg[r.start, :] for i,r in data.iterrows()]

# ? Compute distances matrixes
dist_matrix = np.zeros((len(diffs), len(diffs)))
for i in np.arange(len(diffs)):
    for ii in np.arange(len(diffs)):
        #d = np.sum(calc_distance_between_points_two_vectors_2d(diffs[i], diffs[ii]))
        d = np.sum(np.abs(diffs[i]-diffs[ii]))
        dist_matrix[i, ii] = d

f, ax = plt.subplots()
ax.imshow(dist_matrix)
ax.set(title="distance matrix", xlabel="trial #", ylabel="trial #")

# %%
# Clustering
Z = linkage(distance.squareform(dist_matrix))

#%%
f, ax = plt.subplots()
dendo = dendrogram(Z)

max_d = 2
ax.axhline(max_d, color=grey, lw=2, ls="--")

clust = fcluster(Z, max_d, criterion="distance")
print(len(clust), clust)


ax.set(title="dendogram", xlabel="trial idx", ylabel="distance")


#%%
f, ax = plt.subplots()

colors = get_n_colors(max(clust)+1)
for d, c in zip(diffs, clust):
    ax.plot(d, color=colors[c])

#%%
