
# coding: utf-8

# <h1>Supervised Learning (Classification)</h1>
# <ul>
#     <li>Decision Tree</li>
#     <li>Random Forest</li>
#     <li>Support Vector Machine</li>
# </ul>

# <h1> Datasets </h1>
# <ul>
#     <li>Iris(classification)</li>
# </ul>

# In[1]:


from sklearn.datasets import load_iris
iris_data = load_iris()


# <h2>Iris</h2>

# In[2]:


print (iris_data.DESCR)


# In[3]:


print (iris_data.target)
print ("=====================")
print (iris_data.target_names)
print ("=====================")
print (iris_data.data)
print ("=====================")
print (iris_data.feature_names)


# <h2>Some problems before applying algorithms</h2>
# <ul>
#     <li>Feature Engineering</li>
#     <li>Cross Validation</li>
#     <li>Imbalanced Data</li>
# </ul>

# <h2>Cross Validation</h2>
# <li>Leave One(p) Out Cross Validation (LOOCV)</li>
# <li>k-fold Cross Validation</li>
# 
# <p style='color:red'>sklearn.model_selection</p>

# <h3>Leave One Out Cross Validation</h3>

# In[4]:


from sklearn.model_selection import LeaveOneOut
#in sklearn most of time we use x to present training data
train_data = load_iris().data
#in sklearn most of time we use y to present label
labels = load_iris().target
loocv = LeaveOneOut()
print (loocv.get_n_splits(train_data))


# In[5]:


for train_index, test_index in loocv.split(train_data):
    print (train_index, test_index)


# In[6]:


for train_index, test_index in loocv.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    print (len(X_train), len(X_test), len(Y_train), len(Y_test))


# <h3>K-Fold Cross Validation</h3>

# In[7]:


from sklearn.model_selection import KFold

five_fold = KFold(n_splits=5)
print (five_fold.get_n_splits(train_data))


# In[8]:


for train_index, test_index in five_fold.split(train_data):
    print (train_index, test_index)


# In[9]:


for train_index, test_index in five_fold.split(train_data):
    X_train, X_test = train_data[train_index], train_data[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    print (len(X_train), len(X_test), len(Y_train), len(Y_test))


# <h2>Imbalanced Data</h2>
# <h3>https://imbalanced-learn.org/en/stable/index.html</h3>
# <li>Under Sampling</li>
# <ul>
#     <li>Randomly select the samples from the majority class</li>
#     <li>keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class</li>
# </ul>
# <li>Over Sampling</li>
# <ul>
#     <li>Randomly select the sample from minority class </li>
#     <li>Synthetic Minority Over-sampling Technique (SMOTE) </li>
# </ul>

# In[10]:


from collections import Counter
import numpy as np

labels = load_iris().target
print (np.count_nonzero(labels == 0))
print (np.count_nonzero(labels == 1))
print (np.count_nonzero(labels == 2))
print (Counter(labels))


# In[11]:


new_train_data = load_iris().data[:60]
new_labels = load_iris().target[:60]
print (new_labels)
print (Counter(new_labels))


# <h3> Under Sampling</h3>

# In[12]:


from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids

rus_data, rus_labels = RandomUnderSampler().fit_resample(new_train_data, new_labels)
print (Counter(rus_labels))

cc_data, cc_labels = ClusterCentroids().fit_resample(new_train_data, new_labels)
print (Counter(cc_labels))


# <h3>Over Sampling</h3>

# In[13]:


from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC

ros_data, ros_labels = RandomOverSampler().fit_resample(new_train_data, new_labels)
print (Counter(ros_labels))

sm_data, sm_labels = SMOTE().fit_resample(new_train_data, new_labels)
print (Counter(sm_labels))


# <h2>Decision Tree</h2>
# <ul>
#     <li>sklearn.tree.DecisionTreeClassifier</li>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>
# </ul>

# criterion: Measure the quality of a split.(gini, entropy)<br>
# max_depth: The maximum depth of the tree.<br>
# max_features: The number of features to consider when looking for the best split.<br>

# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:


decision_clf = DecisionTreeClassifier()
decision_clf.fit(iris_data.data,iris_data.target)


# In[16]:


from sklearn.model_selection import cross_val_score
decision_clf = DecisionTreeClassifier()
decision_score = cross_val_score(decision_clf, iris_data.data, iris_data.target, cv=5)
print (decision_score)


# In[17]:


five_fold = KFold(n_splits=5)
score = list()
for train_index, test_index in five_fold.split(iris_data.data, iris_data.target):
    decision_clf = DecisionTreeClassifier()
    decision_clf.fit(iris_data.data[train_index], iris_data.target[train_index])
    score.append(decision_clf.score(iris_data.data[test_index], iris_data.target[test_index]))
print (score)


# In[18]:


clf = DecisionTreeClassifier()
clf.fit(iris_data.data,iris_data.target)
print (clf.predict(iris_data.data[:10]))
print (clf.predict_proba(iris_data.data[:10]))
print (clf.classes_)


# <h2>Random Forest (Ensemble)</h2>
# <ul>
#     <li>sklearn.ensemble.RandomForestClassifier</li>
#     <li>input</li>
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>
# </ul>

# n_estimators: The number of trees in the forest<br>
# criterion: gini, entropy<br>
# max_depth: The maximum depth of the tree<br>
# max_features: The number of features to consider when looking for the best split<br>

# In[19]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(iris_data.data,iris_data.target)


# In[20]:


print (rf_clf.predict(iris_data.data[:50]))
print (rf_clf.predict_proba(iris_data.data[:50]))


# In[21]:


rf_clf = RandomForestClassifier()
rf_score = cross_val_score(rf_clf, iris_data.data, iris_data.target, cv=5)
print (rf_score)


# <h2>Support Vector Machine</h2>
# <ul>
#     <li>sklearn.svm.SVC</li>
#     <li>input
#     <ul>
#         <li>Training Data</li>
#         <li>Label</li>
#     </ul>
# <ul>

# C<br>
# &nbsp;&nbsp;&nbsp;Penalty parameter<br>
# kernel<br>
# &nbsp;&nbsp;&nbsp;linear, polynomial, rbf, sigmoid<br>
# gamma<br>
# &nbsp;&nbsp;&nbsp;Kernel coefficient 

# In[22]:


from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(iris_data.data,iris_data.target)
print (svm_clf.predict(iris_data.data[:50]))


# In[23]:


svm_clf = SVC(probability=True)
svm_clf.fit(iris_data.data,iris_data.target)
print (svm_clf.predict_proba(iris_data.data[:50]))
print (svm_clf.classes_)


# In[24]:


svm_clf = SVC()
svm_score = cross_val_score(clf, iris_data.data, iris_data.target, cv=5)
print (svm_score)


# <h2>Kaggle Example: Titanic</h2>
# <p>https://www.kaggle.com/c/titanic</p>
# <ul>
# <li>Training Data (With Label)</li>
# <li>Unseen Data (Without Label)</li>
# <li>Other Informations</li>
# </ul>

# In[25]:


import csv

header = list()
content = list()
with open('train.csv','r') as f:    
    csv_write = csv.reader(f)
    for line in csv_write:
        content.append(line)
    header = content.pop(0)
    
print (header)
print (len(content))
print (content)


# In[26]:


#select pclass, gender as features

training_data = list()
labels = [ row[1] for row in content]

for index,row in enumerate(content):
    pclass = int(row[2])
    gender = 1 if row[4] == 'male' else 0
    training_data.append([pclass, gender])
    

print (training_data)
print (labels)


# In[27]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from statistics import mean

#support vector machine
svm_clf = SVC(C=8,gamma=0.125)
svm_score = cross_val_score(svm_clf,training_data,labels,cv=5)

#random forest
rf_clf = RandomForestClassifier()
rf_score = cross_val_score(rf_clf,training_data,labels,cv=5)

#decision tree
decision_clf = DecisionTreeClassifier()
decision_score = cross_val_score(decision_clf,training_data,labels,cv=5)

print (mean(svm_score))
print (mean(rf_score))
print (mean(decision_score))


# In[28]:


rf_clf.fit(training_data,labels)


# In[29]:


import csv
header_test = list()
content_test = list()
with open('test.csv','r') as f:
    
    csv_write = csv.reader(f)
    for line in csv_write:
        content_test.append(line)
    header_test = content_test.pop(0)
    
print (header_test)
print (len(content_test))
print (content_test)


# In[30]:


testing_data = list()


for index,row in enumerate(content_test):
    pclass = int(row[1])
    gender = 1 if row[3] == 'male' else 0
    testing_data.append([pclass, gender])

print (testing_data)


# In[31]:


predictions = rf_clf.predict(testing_data)
print (predictions)


# In[32]:


with open('./result.csv','w') as f:
    f.write("PassengerId,Survived\n")
    for index, prediction in enumerate(predictions):
        output_str = str(content_test[index][0]) + "," + str(prediction) + "\n"
        f.write(output_str)

