
# coding: utf-8

# <h1>Unsupervised Learning</h1>
# <ul>
#     <li>K-means</li>
#     <li>Hierarchical Clustering</li>
#     <li>GMM</li>
#     <li>DBScan</li>
#     
# </ul>

# <h1>Dataset</h1>
# <ul>
#     <li>Wine Dataset</li>
# </ul>

# In[1]:


from sklearn.datasets import load_wine
wine_data = load_wine()


# <h2>Wine Dataset</h2>

# In[2]:


print (wine_data.DESCR)


# In[3]:


print (wine_data.data)


# <h2>Types of Input Data</h2>
# <ul>
#     <li>One Mode Matrix</li>
#     <li>Two Mode Matrix</li>
# </ul>

# <h2>Dissimilarity Matrix ( One mode matrix)</h2>
# <ul>
# <li>object by object</li>
# <li> n * n matrix</li>    
# </ul>

# How do we transform two mode data into one mode data<br>

# <p>L1 (manhattan) $|x_{1}-x_{2}|+|y_{1}-y_{2}|.$</p>

# <p>L2 (euclidean) $\sqrt{(x_{1}-x_{2})^{2}+(y_{1}-y_{2})^{2}}.$</p>

# <p>cosine $\frac {x \cdot y}{||x|| \cdot ||y||}$</p>

# <p style="color:red">we can use sklearn.metrics.pairwise</p>

# In[4]:


from sklearn.metrics import pairwise
#manhattan
one_mode_L1_data = pairwise.manhattan_distances(load_wine().data,load_wine().data)
#euclidean
one_mode_L2_data = pairwise.euclidean_distances(load_wine().data,load_wine().data)




# In[5]:


print (one_mode_L1_data.shape)
print (one_mode_L1_data[0])
print (one_mode_L2_data[0])


# <h2>Data Matrix (Two mode Matrix)</h2>
#     <li>n objects with p attributes</li>
#     <li>n * p matrix</li>
# 

# In[6]:


two_mode_data = load_wine().data
print (two_mode_data)
print (two_mode_data.shape)


# <h2>K-Means</h2>
# <ul>
#     <li>sklearn.cluster.Kmeans</li>
#     <li>input
#     <ul>
#         <li>k clusters</li>
#         <li>data (n object with p attribute)</li>
#     </ul>
# </ul>

# n_clusters<br>
# &nbsp;&nbsp;&nbsp;the number of clusters to form as well as the number of centroids to generate<br>
# random_state

# In[7]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
kmeans_data = MinMaxScaler().fit_transform(two_mode_data)
kmeans = KMeans(n_clusters=3, random_state=0).fit(kmeans_data)
print (kmeans.labels_)


# In[8]:


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
result = model.fit_transform(kmeans_data)
print (result)


# In[9]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# Get current size
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 4.0
fig_size[1] = 3.0
plt.rcParams["figure.figsize"] = fig_size
        



# In[10]:


result_0 = np.array([result[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0])
result_1 = np.array([result[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1])
result_2 = np.array([result[i] for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 2])

plt.scatter(result_0[:,0], result_0[:,1], c='y', linewidths=0.5, s=10)
plt.scatter(result_1[:,0], result_1[:,1], c='g', linewidths=0.5, s=10)
plt.scatter(result_2[:,0], result_2[:,1], c='b', linewidths=0.5, s=10)
flg_kmeans = plt


# <h2>Hierarchical Clustering</h2>
# <ul>
#     <li>sklearn.cluster.AgglomerativeClustering</li>
#     <li>input</li>
#     <ul>
#         <li>data (n objects by n objects)</li>
#     </ul>
# </ul>

# n_clusters<br>
# &nbsp;&nbsp;&nbsp;The number of clusters to find<br>
# affinity<br>
# &nbsp;&nbsp;&nbsp;Can be 'euclidean'(default), 'l1', 'l2', 'manhattan', 'cosine' or 'precomputed'<br>
# linkage<br>
# &nbsp;&nbsp;&nbsp;'ward', 'complete', 'average'
# <p style="color:red">If linkage is “ward”, only “euclidean” is accepted</p>

# In[11]:


from sklearn.cluster import AgglomerativeClustering

L2_hierarchical_1 = AgglomerativeClustering(n_clusters=6, linkage='complete', affinity='l2').fit(two_mode_data)
L2_hierarchical_2 = AgglomerativeClustering(n_clusters=6, linkage='complete',affinity='precomputed').fit(one_mode_L2_data)
print (L2_hierarchical_1.labels_)
print (L2_hierarchical_2.labels_)


# In[12]:


from sklearn.preprocessing import MinMaxScaler
hierarchical_data = MinMaxScaler().fit_transform(two_mode_data)
hierarchical_example = AgglomerativeClustering(n_clusters=3, linkage='complete').fit(hierarchical_data)


# In[13]:


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
result = model.fit_transform(hierarchical_data)
print (result)


# In[14]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# Get current size
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 4.0
fig_size[1] = 3.0
plt.rcParams["figure.figsize"] = fig_size


# In[15]:


result_0 = np.array([result[i] for i in range(len(hierarchical_example.labels_)) if hierarchical_example.labels_[i] == 0])
result_1 = np.array([result[i] for i in range(len(hierarchical_example.labels_)) if hierarchical_example.labels_[i] == 1])
result_2 = np.array([result[i] for i in range(len(hierarchical_example.labels_)) if hierarchical_example.labels_[i] == 2])

plt.scatter(result_0[:,0], result_0[:,1], c='y', linewidths=0.5, s=10)
plt.scatter(result_1[:,0], result_1[:,1], c='g', linewidths=0.5, s=10)
plt.scatter(result_2[:,0], result_2[:,1], c='b', linewidths=0.5, s=10)
flg_hierarchical = plt


# <h2>Gaussian Mixture Model</h2>
# <ul>
# <li>sklearn.mixture.GaussianMixture
# <li>input
#     <ul>
#         <li>k clusters
#         <li>data (n object with p attributes)</li>
#     </ul>
# </ul>

# n_components<br>
# &nbsp;&nbsp;&nbsp;The number of mixture components<br>
# covariance_type<br>
# &nbsp;&nbsp;&nbsp;‘full’, ‘tied’, ‘diag’, ‘spherical’<br> 
# random_state<br>

# In[16]:


from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

gmm_data = MinMaxScaler().fit_transform(two_mode_data)
gmm = GaussianMixture(n_components=3,random_state=0)
gmm.fit(gmm_data)
gmm_labels = gmm.predict(gmm_data)
print (gmm_labels)


# In[17]:


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
result = model.fit_transform(gmm_data)
print (result)


# In[18]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# Get current size
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 4.0
fig_size[1] = 3.0
plt.rcParams["figure.figsize"] = fig_size


# In[19]:


result_0 = np.array([result[i] for i in range(len(gmm_labels)) if gmm_labels[i] == 0])
result_1 = np.array([result[i] for i in range(len(gmm_labels)) if gmm_labels[i] == 1])
result_2 = np.array([result[i] for i in range(len(gmm_labels)) if gmm_labels[i] == 2])

plt.scatter(result_0[:,0], result_0[:,1], c='y', linewidths=0.5, s=10)
plt.scatter(result_1[:,0], result_1[:,1], c='g', linewidths=0.5, s=10)
plt.scatter(result_2[:,0], result_2[:,1], c='b', linewidths=0.5, s=10)
flg_gmm = plt


# <h2>DBScan</h2>
# <ul>
# <li>sklearn.cluster.DBSCAN
# <li>input
#     <ul>
#         <li>Eps: Maximum radius of the neighborhood</li>
#         <li>MinPts: Minimum number of points in an Eps-neighborhood of that point</li>
#         <li>data (n object with p attributes)</li>
#     </ul>
# </ul>

# eps<br>
# &nbsp;&nbsp;&nbsp;The maximum distance between two samples for them to be considered as in the same neighborhood<br>
# min_samples<br>
# &nbsp;&nbsp;&nbsp;The number of samples (or total weight) in a neighborhood for a point to be considered as a core point

# In[20]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=30, min_samples=9)
dbscan.fit(two_mode_data)
dbscan_labels = dbscan.labels_
print (dbscan_labels)


# In[21]:


n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print (n_clusters)


# In[22]:


for index in range(n_clusters):
    print (index, ":", dbscan_labels.tolist().count(index))
print ("outlier : ", dbscan_labels.tolist().count(-1))


# In[23]:


from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=0)
result = model.fit_transform(two_mode_data)
print (result)


# In[24]:


from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# Get current size
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 8.0
fig_size[1] = 6
.0
plt.rcParams["figure.figsize"] = fig_size


# In[25]:


result_0 = np.array([result[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 0])
result_1 = np.array([result[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 1])
result_2 = np.array([result[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 2])
result_3 = np.array([result[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 3])
outlier = np.array([result[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == -1])


plt.scatter(result_0[:,0], result_0[:,1], c='y', linewidths=0.5, s=5)
plt.scatter(result_1[:,0], result_1[:,1], c='g', linewidths=0.5, s=5)
plt.scatter(result_2[:,0], result_2[:,1], c='b', linewidths=0.5, s=5)
plt.scatter(result_3[:,0], result_3[:,1], c='m', linewidths=0.5, s=5)
plt.scatter(outlier[:,0], outlier[:,1], c='r', linewidths=0.5, s=10)
flg_gmm = plt

