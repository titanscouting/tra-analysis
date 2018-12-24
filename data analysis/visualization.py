#Titan Robotics Team 2022: Visualization Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#   this should be imported as a python module using 'import visualization'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#Number of easter eggs: Jake is Jewish and does not observe easter.
#setup:

__version__ = "1.0.0.001"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.0.0.xxx:
    -added basic plotting, clustering, and regression comparisons"""
__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
    )
__all__ = [
    'affinity_prop',
    'bar_graph',
    'dbscan',
    'kmeans',
    'line_plot',
    'pca_comp',
    'regression_comp',
    'scatter_plot',
    'spectral',
    'vis_2d'
    ]
#imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, SpectralClustering

#bar of x,y
def bar_graph(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    plt.bar(x,y)
    plt.show()

#scatter of x,y
def scatter_plot(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    plt.scatter(x,y)
    plt.show()

#line of x,y
def line_plot(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    plt.scatter(x,y)
    plt.show()

#plot data + regression fit
def regression_comp(x,y,reg):
    x=np.asarray(x)
    y=np.asarray(y)
    regx=np.arange(x.min(),x.max(),(x.max()-x.min())/1000)
    regy=[]
    for i in regx:
        regy.append(eval(reg[0].replace("z",str(i))))
    regy=np.asarray(regy)
    plt.scatter(x,y)
    plt.plot(regx,regy,color="orange",linewidth=3)
    plt.text(.85*max([x.max(),regx.max()]),.95*max([y.max(),regy.max()]),
            u"R\u00b2="+str(round(reg[2],5)),
            horizontalalignment='center', verticalalignment='center')
    plt.text(.85*max([x.max(),regx.max()]),.85*max([y.max(),regy.max()]),
            "MSE="+str(round(reg[1],5)),
             horizontalalignment='center', verticalalignment='center')
    plt.show()

#PCA to compress down to 2d
def pca_comp(big_multidim):
    pca=PCA(n_components=2)
    td_norm=StandardScaler().fit_transform(big_multidim)
    td_pca=pca.fit_transform(td_norm)
    return td_pca

#one-stop visualization of multidim datasets
def vis_2d(big_multidim):
    td_pca=pca_comp(big_multidim)
    plt.scatter(td_pca[:,0], td_pca[:,1])

def cluster_vis(data, cluster_assign):
    pca=PCA(n_components=2)
    td_norm=StandardScaler().fit_transform(data)
    td_pca=pca.fit_transform(td_norm)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(clu) + 1))))
    colors = np.append(colors, ["#000000"])
    plt.figure(figsize=(8, 8))
    plt.scatter(td_norm[:, 0], td_norm[:, 1], s=10, color=colors[cluster_assign])
    plt.show()

#affinity prop- slow, but ok if you don't have any idea how many you want
def affinity_prop(data, damping=.77, preference=-70):
    td_norm=StandardScaler().fit_transform(data)
    db = AffinityPropagation(damping=damping,preference=preference).fit(td)
    y=db.predict(td_norm)
    return y

#DBSCAN- slightly faster but can label your dataset as all outliers
def dbscan(data, eps=.3):
    td_norm=StandardScaler().fit_transform(data)
    db = DBSCAN(eps=eps).fit(td)
    y=db.labels_.astype(np.int)
    return y

#K-means clustering- the classic
def kmeans(data, num_clusters):
    td_norm=StandardScaler().fit_transform(data)
    db = KMeans(n_clusters=num_clusters).fit(td)
    y=db.labels_.astype(np.int)
    return y

#Spectral Clustering- Seems to work really well
def spectral(data, num_clusters):
    td_norm=StandardScaler().fit_transform(data)
    db = SpectralClustering(n_clusters=num_clusters).fit(td)
    y=db.labels_.astype(np.int)
    return y
