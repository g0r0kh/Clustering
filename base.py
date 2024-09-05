import pandas as pd
import numpy as np
from louis import plain_text
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from matplotlib import pyplot as plt
from matplotlib import rc
from sklearn.cluster import KMeans

#
# xlsx source import
data = pd.read_excel('sample_data.xlsx',
                     sheet_name='sheet1')

print(data[:4])