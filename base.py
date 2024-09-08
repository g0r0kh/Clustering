import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.cluster import KMeans



df= pd.read_excel('sample_data.xlsx',
                     sheet_name='sheet1')

col = ['ССЧ', 'Выручка']

pd.options.mode.chained_assignment = None
df[col].fillna(0, inplace=True)
#
# from pandas.plotting import scatter_matrix
# scatter_matrix(df[col], alpha=0.05, figsize=(10, 10))
# plt.show()
df[col].corr() #

#
from sklearn import preprocessing
dataNorm = preprocessing.MinMaxScaler().fit_transform(df[col].values)

data_dist = pdist(dataNorm, 'euclidean')
data_linkage = linkage(data_dist, method='average')


last = data_linkage[-10:, 2]
last_rev = last[::-1]

plt.figure(figsize=(10, 8))
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.grid()
plt.show()
# k = acceleration.argmax()+2
# print("Рекомендованное количество кластеров:", k)
# positive acceleration count
# print("Recommend clasters count:", len(acceleration_rev[acceleration_rev>0]))
print("Recommend clasters count:", len(acceleration_rev))

def fancy_dendrogram(*args, **kwargs):
    plt.figure(figsize=(10, 8))
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

nClust=len(acceleration_rev)


fancy_dendrogram(
    data_linkage,
    truncate_mode='lastp',
    p=nClust,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
)
plt.grid()
plt.show()

#
clusters=fcluster(data_linkage, nClust, criterion='maxclust')

x=0 #
y=1 #
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=clusters, cmap='flag')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.grid()
plt.show()
#

df['I']=clusters
res=df.groupby('I')[col].mean()
res['Количество']=df.groupby('I').size().values




# # KMeans
km = KMeans(n_clusters=nClust).fit(dataNorm)

km.labels_ +1

x=0 #
y=1 #
centroids = km.cluster_centers_
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=km.labels_, cmap='flag')
plt.scatter(centroids[:, x], centroids[:, y], marker='+', s=300,
            c='y', label='centroid')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.grid()
plt.show()
#


df['KMeans']=km.labels_+1



# #
df.to_excel('result_claster.xlsx', index=False)