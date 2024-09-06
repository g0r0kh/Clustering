import pandas as pd
import numpy as np
# from louis import plain_text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.cluster import KMeans


data = pd.read_excel('sample_data.xlsx',
                     sheet_name='sheet1')

# data.info()
# data.describe()
# data.columns()

col = ['ССЧ', 'Выручка']

# from pandas.plotting import scatter_matrix
# scatter_matrix(data[col], alpha=0.05, figsize=(10, 10))
# plt.show()

# data[col].corr()
from sklearn import preprocessing
dataNorm = preprocessing.MinMaxScaler().fit_transform(data[col].values)

data_dist = pdist(dataNorm, 'euclidean')
data_linkage = linkage(data_dist, method='average')

last = data_linkage[-10:, 2]
last_rev = last[::-1]


idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)
acceleration_rev = acceleration[::1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.grid()
plt.show()

k = acceleration_rev.argmax() + 2
print("Рекомендованное количество кластеров:", k)

#функция построения дендрограмм
def fancy_dendrogram(*args, **kwargs):
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

nClust=9

#строим дендрограмму
fancy_dendrogram(
    data_linkage,
    truncate_mode='lastp',
    p=nClust,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
)
plt.show()


# иерархическая кластеризация
clusters=fcluster(data_linkage, nClust, criterion='maxclust')
# print(clusters)

x=0 # Чтобы построить диаграмму в разных осях, меняйте номера столбцов
y=1 #
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=clusters, cmap='flag')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.show()

# к оригинальным данным добавляем номер кластера
data['I']=clusters
res=data.groupby('I')[col].mean()
res['Количество']=data.groupby('I').size().values
print(res) #ниже средние цифры по кластерам и количество объектов (Количество)