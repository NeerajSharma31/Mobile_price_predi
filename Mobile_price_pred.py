#!/usr/bin/env python
# coding: utf-8

# # Mobile price Prediction. 

# Predict a price range, indicating how high the price is, using different Machine Learning algorithms.

# ### Step1. Importing the libraries

# In[20]:


import numpy as np
import pandas as pd


# ### Step2. Creating & Reading the data

# In[21]:


#Creating the data
mobile_data = pd.read_csv(r"C:\Users\neeraj nandkumar\Downloads\Mobile_data.csv")


# In[22]:


print(mobile_data.shape)


# In[23]:


mobile_data.head()


# In[24]:


mobile_data.columns


# In[25]:


mobile_data=mobile_data[['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
       'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
       'sc_w','talk_time', 'price_range']]


# In[26]:


print(mobile_data.shape)


# The columns which have been given in the dataset are important & can't be dropped or deleted from dataset, as all the columns provided are in relation to the price predictions of mobile.
# 

# ### Step3. Performing EDA 

# In[27]:


mobile_data.describe(include='all')


# In[28]:


mobile_data.info()


# In[29]:


mobile_data.dtypes


# ### Step4. Check if there are missing values. If yes, handle them.
# 

# In[30]:


#Finding missing values
print(mobile_data.isnull().sum())


# In[31]:


#To check if there are any special characters in place of values 
for i in mobile_data.columns:
    print({i:mobile_data[i].unique()})


# From the above observations we can observe that there are no missing values or unique value in the data.
# So,we won't eliminate or treat any missing values

# In[32]:


mobile_data.corr()


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt

corr = mobile_data.corr()
plt.figure(figsize=(20,30))
sns.heatmap(corr, vmin=-1.0,vmax=1.0,annot=True)
plt.yticks(rotation=0)
plt.show()


# In[34]:


sns.set()
price_plot=mobile_data['price_range'].value_counts().plot(kind='bar')
plt.xlabel('price_range')
plt.ylabel('Count')
plt.show()


# Observations:-
# 1. The value of 0 --> low cost
# 2. The value of 1 --> medium cost
# 3. The value of 2 --> high cost
# 4. The value of 3 --> very high cost

# In[35]:


mobile_data.info()


# In[36]:


sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data=mobile_data["battery_power"])

plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data=mobile_data["int_memory"])

plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data=mobile_data["m_dep"])

plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data=mobile_data["pc"])

plt.show()


# In[ ]:


sns.set(rc={'figure.figsize':(5,5)})
ax=sns.displot(data=mobile_data["ram"])

plt.show()


# ### Step5. Creating X & Y

# In[ ]:


#Create X & Y 
X = mobile_data.values[:,0:-1]
Y = mobile_data.values[:,-1]


# In[ ]:


print(X.shape)
print(Y.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X)
X = scaler.transform(X)


# In[ ]:


print(X)


# ## Step6. Train-Test Splitting

# In[ ]:


from sklearn.model_selection import train_test_split  #<1000=in range of 80-20  &  >1000=in range of 70-30

#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[ ]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# # 1. KNN Classifier

# In[ ]:


#predicting using the KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),
                              metric='euclidean')

#euclidean,manhattan,minkowski
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# In[ ]:


from sklearn.model_selection import train_test_split  #<1000=in range of 80-20  &  >1000=in range of 70-30

#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[ ]:


#predicting using the KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),
                              metric='manhattan')

#euclidean,manhattan,minkowski
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# In[ ]:


from sklearn.model_selection import train_test_split  #<1000=in range of 80-20  &  >1000=in range of 70-30

#Split the data into test and train
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)


# In[ ]:


#predicting using the KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X_train))),
                              metric='minkowski')

#euclidean,manhattan,minkowski
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# 1. KNN was build on 3 metrics Euclidean,Manhattan & Minkowski
# 2. From the Observations we can see that the KNN model with "manhattan" metric was showing accuracy then that of the 
#    other metrics.
# 3. So, manhattan metric will be taken into consideration for tuning the model for getting good accuracy & better results

# ### Tuning of the KNN model

# In[ ]:


from sklearn.metrics import accuracy_score
my_dict={}
for K in range(1,40):
    model_KNN=KNeighborsClassifier(n_neighbors=K,metric='manhattan')
    model_KNN.fit(X_train,Y_train)
    Y_pred=model_KNN.predict(X_test)
    print("Accuracy is",accuracy_score(Y_test,Y_pred), "for K-Value:",K)
    my_dict[K]=accuracy_score(Y_test,Y_pred)


# In[ ]:


for k in my_dict:
    if my_dict[k]==max(my_dict.values()):
        print(k,":",my_dict[k])


# In[ ]:


#predicting using the KNeighbors_Classifier
from sklearn.neighbors import KNeighborsClassifier
model_KNN=KNeighborsClassifier(n_neighbors=34, metric='euclidean')

#euclidean,manhattan,minkowski
#fit the model on the data and predict the values
model_KNN.fit(X_train,Y_train)

Y_pred=model_KNN.predict(X_test)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# Tuned KNN model was not showing good results & it was not even giving a good accuracy which is expected

# # 2. Logistic Regression 

# In[ ]:


from sklearn.linear_model import LogisticRegression
#create a model object
classifier = LogisticRegression()

#train the model object
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# Logistic was giving a good accuracy & it was also misclassifying less observations

# # 3. Random Forests

# In[ ]:


#predicting using the Random_Forest_Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest = RandomForestClassifier(n_estimators=50,random_state=10)

#fit the model on the data and predict the values
model_RandomForest.fit(X_train,Y_train)

Y_pred = model_RandomForest.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# # 4. Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

#fit the model on the data and predict the values
gnb.fit(X_train,Y_train)

Y_pred = gnb.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# # 5. SVM Classifier

# In[ ]:


from sklearn.svm import SVC
svc_model=SVC(kernel='rbf',C=20,gamma=0.01)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# # 6. Decision Trees

# In[ ]:


#predicting using the Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier(criterion="gini",random_state=10,
                                           splitter="best",min_samples_leaf=5,
                                           max_depth=10,max_leaf_nodes=10)

#fit the model on the data and predict the values
model_DecisionTree.fit(X_train,Y_train)

Y_pred = model_DecisionTree.predict(X_test)
#print(Y_pred)
#print(list(zip(Y_test,Y_pred)))


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# # 7. XGBoost

# In[ ]:


from xgboost import XGBClassifier
model_XGBoost=XGBClassifier(n_estimators=100,
                              random_state=10)

#fit the model on the data and predict the values
model_XGBoost.fit(X_train,Y_train)

Y_pred=model_XGBoost.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cfm=confusion_matrix(Y_test,Y_pred)
sns.heatmap(cfm, annot=True, fmt='g', cbar=False, cmap='BuPu')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


print("Classification report:")

print(classification_report(Y_test,Y_pred))

acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model: ",acc)


# From all the models we applied we saw that the logistic regression was showing more accuracy & the misclassification of the values was in very less in number as compared to the other models.

# In[ ]:




