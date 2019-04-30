
# coding: utf-8

# # NetworkX 

# NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks
# 

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
import networkx as nx


# #### test if networkx is working

# In[2]:


import networkx as nx
print (nx)
print (nx.__version__)


# ## basic operations

# In[3]:


G = nx.Graph()     # build undirected graph
G.add_node(1)      # add a node named "1"(1 is the label)
G.add_edge(2,3)    # add an edge from node2 to node3 
                   #  if node2 and node3 do not exist, it will auto add node2 and node3
G.add_edge(3,2)    # add an edge from node3 to node2

print ('Nodes:', G.nodes)
print ('Edges:', G.edges)

G.add_edge(1,3)
G.add_edge(2,4)
G.add_edge(3,4)

print ("=================")
print ('Nodes:', G.nodes)
print ('Edges:', G.edges)


# Graph(): Undirected Graph<br>
# DiGraph(): Directed Graph<br>
# MultiGraph(): Multiedges Undirected Graph<br>
# MultiDiGraph(): Multiedges Directed Graph<br>

# In[4]:


print (G.nodes())
print (G.edges())
print (G.number_of_edges())
print (G.number_of_nodes())
print (list(G.adj[2]))
print (G.degree(2))


# ## use draw() function in networkx to plot the graph figure

# node_size   :  the size of all nodes<br>
# node_color  :  the color of all nodes<br>
# node_shape  :  the shape of all nodes<br>
# alpha       :  transparency<br>
# width       :  width of edge<br>
# edge_color  :  color of edge<br>
# style       :  style of edge<br>
# with_labels :  node with label<br>
# font_size   :  size of label<br>
# font_color  :  color of label<br>

# In[5]:


#disable the warnings
import warnings
warnings.filterwarnings("ignore")

nx.draw(G,with_labels=True)


# ### Layout

# circular_layout： Position nodes on a circle<br>
# random_layout  ： Position nodes uniformly at random in the unit square<br>
# shell_layout   ： Position nodes in concentric circles<br>
# spring_layout  ： Position nodes using Fruchterman-Reingold force-directed algorithm<br>
# spectral_layout： Position nodes using the eigenvectors of the graph Laplacian<br>

# In[6]:


#setting
pos_cir = nx.circular_layout(G)
#draw
nx.draw(G,pos_cir,with_labels=True)


# In[7]:


#setting
pos_cir = nx.random_layout(G)
#draw
nx.draw(G,pos_cir,with_labels=True)


# ### Another way to add nodes and edges

# In[8]:


G = nx.Graph()
G.add_nodes_from([1,2,3,4,5])
G.add_edges_from([(4,5), (4,2)])


# In[9]:


print (G.nodes())
print (G.edges())


# ### Path_Graph

# In[10]:


H = nx.path_graph(10)

print (H.nodes)
print (H.edges)
print (H[0], H[1], H[2])


# ### Edge Weight

# In[11]:


G = nx.Graph()
G.add_nodes_from([1,2,3])
G.add_edge(2,3)
print (G.edges)


# In[12]:


G.add_edge(1,2,weight=3.1415)
print (G[1], G[2])


# ### Remove & Clear

# In[13]:


G = nx.Graph()
G.add_nodes_from([1,2,3,4])
G.add_edge(1,3)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(2,4)
print (G.edges)


# In[14]:


G.remove_node(4)
print (G.edges)


# In[15]:


G.remove_edge(2,3)
print (G.nodes)
print (G.edges)


# In[16]:


G.clear()
print (G.nodes)
print (G.edges)


# ### Directed Graphs

# In[17]:


G = nx.Graph()
G.add_node(1)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,2)

DG = nx.DiGraph()
DG.add_node(1)
DG.add_edge(1,2)
DG.add_edge(2,3)
DG.add_edge(3,2)

print (G[1], G[2], G[3])
print (DG[1], DG[2], DG[3])


# In[18]:


DG = nx.DiGraph()
DG.add_weighted_edges_from([(1,2,0.5),(3,1,0.75),(1,4,0.4)])
print (DG[1], DG[2], DG[3], DG[4])


# In[19]:


print (DG.degree(1))   #outdegree:2 indegree:1
print (DG.out_degree(1))
print (DG.in_degree(1))
print (DG.out_degree(1,weight='weight'))  #the sum of the weight of node1's out degree
print (DG.degree(1,weight='weight'))      #the sum of the weight of node1'a out degree + in degree
print ([n for n in DG.neighbors(1)])
print (DG.neighbors(1))


# In[20]:


nx.draw(DG,with_labels=True)


# ### Accessing nodes

# #### Use adjacency_iter() to visit all the node in graph<br> return (node,adjacencydict)

# In[21]:


G = nx.Graph()
G.add_path([0,1,2,3])
print (G.edges)


# In[22]:


for n,nbrdict in G.adjacency():
    print (n,nbrdict)


# ### Save your graph

# In[23]:


import pickle
G = nx.path_graph(10)
pickle.dump(G,open('networkx_example.txt','wb'))


# In[24]:


G_load = pickle.load(open('networkx_example.txt', 'rb'))
print (G_load.edges)


# ## Example: DFS

# In[25]:


G = nx.Graph()  
G.add_edges_from([(1,5), (1,2), (2,3), (2,4), (3,4), (3,5), (4,5), (5,3)])  
stack = [1]  # Start from node1  
visit_list = []  
nx.draw(G, with_labels=True)


# In[26]:


while len(stack) > 0:  
    vnode = stack.pop()  
    if vnode not in visit_list:  
        print("\t[Info] Visit {0}...".format(vnode))  
        visit_list.append(vnode)  
    nbs = G.neighbors(vnode)  
    for nb in nbs:  
        if nb not in visit_list:  
            print("\t[Info] Put {0} in stack...".format(nb))  
            stack.append(nb)  
    print("\tStack list={0}".format(stack))  
    print("\tVisit list={0}".format(visit_list))  
    if len(visit_list) == len(G.nodes()): break  
  
print("\t[Info] Deep First Search has {0}".format(visit_list))  


# ## Social Network Measure

# ### Shorest Path

# In[27]:


G = nx.Graph()
G.add_edges_from([(1,5), (1,2), (2,3), (2,4), (3,4), (3,5), (4,5)])  
nx.draw(G,with_labels=True)


# In[28]:


p = nx.shortest_path(G)
print (p[1][3])
print (p[2][4])
print (p[1][4])


# ### Degree Centrality

# In[29]:


G = nx.Graph()
G.add_edges_from([(1,5), (1,2), (2,3), (2,4), (3,4), (3,5), (4,5)])  
nx.draw(G,with_labels=True)


# In[30]:


DG = nx.DiGraph()
DG.add_edges_from([(1,5), (1,2), (2,3), (2,4), (3,4), (3,5), (4,5)])
nx.draw(DG,with_labels=True)


# In[31]:


#degree centrality
d_c = nx.degree_centrality(G)
print (d_c)


# In[32]:


#in-degree centrality
in_dc = nx.in_degree_centrality(DG)
print (in_dc)
#out-degree centrality
out_dc = nx.out_degree_centrality(DG)
print (out_dc)


# ### Betweenness Centrality

# In[33]:


b_c = nx.betweenness_centrality(G)
print (b_c)


# ### Closeness_centrality

# In[34]:


c_c = nx.closeness_centrality(G)
print (c_c)


# ### Eigenvector_centrality

# In[35]:


e_c = nx.eigenvector_centrality(G)
print (e_c)


# ### Clique

# In[36]:


nx.draw(G,with_labels=True)
cliques = list(nx.enumerate_all_cliques(G))
print (cliques)
for i in range(1,len(max(cliques, key=len))+1):
    print (i,"clique:", [x for x in cliques if len(x)==i])


# In[37]:


node_cliques = nx.cliques_containing_node(G,nodes=1)
print(node_cliques)


# ### Triangle, Transitivity, Clustering Coefficient

# In[38]:


print (nx.triangles(G))
print (nx.transitivity(G))
print (nx.clustering(G))


# ## Comunity

# https://github.com/taynaud/python-louvain

# In[39]:


import community

G = nx.Graph()
G.add_edges_from([(1,5), (1,2), (2,4), (3,4), (3,5), (4,5)])  
partition = community.best_partition(G)
print (partition)


# In[40]:


size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
labels = {}    
for node in G.nodes():
    labels[node] = node
for com in set(partition.values()) :
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20, node_color = str(com / size))
nx.draw_networkx_edges(G,pos, alpha=0.5)
nx.draw_networkx_labels(G,pos,labels)


# In[41]:


print (community.modularity(partition,G))

