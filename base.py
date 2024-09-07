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

from pandas.plotting import scatter_matrix
scatter_matrix(data[col], alpha=0.05, figsize=(10, 10))
plt.show()

data[col].corr()
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
plt.grid()
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

# строим кластеризаци методом KMeans
km = KMeans(n_clusters=nClust).fit(dataNorm)

# выведем полученное распределение по кластерам
# так же номер кластера, к котрому относится строка, так как нумерация начинается с нуля, выводим добавляя 1
km.labels_ +1

x=0 # Чтобы построить диаграмму в разных осях, меняйте номера столбцов
y=1 #
centroids = km.cluster_centers_
plt.figure(figsize=(10, 8))
plt.scatter(dataNorm[:,x], dataNorm[:,y], c=km.labels_, cmap='flag')
plt.scatter(centroids[:, x], centroids[:, y], marker='*', s=300,
            c='r', label='centroid')
plt.xlabel(col[x])
plt.ylabel(col[y])
plt.show()



# к оригинальным данным добавляем номера кластеров
data['KMeans']=km.labels_+1
res=data.groupby('KMeans')[col].mean()
res['Количество']=data.groupby('KMeans').size().values
res

data[data['KMeans']==6] # изменяйте номер кластера, содержание которого хотите просмотреть

# data[data['KMeans']==4][['Review', 'Star', 'ordersCount', 'Value', 'brandName', 'goodsName',
#        'Мощность устройства', 'Объем чайника', 'Цвет']]

# сохраним результаты в файл
data.to_excel('result_claster.xlsx', index=False)