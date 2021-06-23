#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and downloading data

# In[ ]:


from google.colab import files
files.upload()


# In[2]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
# change permission
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[3]:


get_ipython().system('kaggle datasets download -d mlg-ulb/creditcardfraud')


# In[4]:


get_ipython().system('unzip "/content/creditcardfraud.zip" -d "/content/"')


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[2]:


df = pd.read_csv("creditcard.csv", delimiter = ",")
df.head()


# In[3]:


df["Class"].value_counts()


# In[4]:


print("dataset contains {}% for non-fradulent class".format(df["Class"].value_counts()[0]*100/len(df["Class"])))


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.drop(["Time"], axis=1, inplace = True)
df.head()


# In[8]:


corr = df.corr("pearson")
plt.figure(figsize = (10,10))
sns.heatmap(corr, linewidth = 1)


# In[9]:


plt.figure(figsize = (10, 10))
sns.heatmap(abs(corr) > 0.5, linewidth = 1)


# In[10]:


x, y = df.iloc[:, :-1], df.iloc[:, -1]
x.shape, y.shape


# In[11]:


x = np.array(x, dtype = "float")
y = np.array(y, dtype = "float")


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y, test_size = .25)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Random Forest without SMOTE

# In[13]:


rf_model = RandomForestClassifier(max_depth = None, n_jobs = -1)
params = {"n_estimators": [10, 50, 100, 250], 
          "max_features": [.1, .5, .9] 
          }

gridsearch = GridSearchCV(rf_model, params, cv = 3, n_jobs = -1, return_train_score = True, scoring = "f1", verbose = 10)
gridsearch.fit(x_train, y_train)


# In[14]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results.head()


# In[16]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_train_score")
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_features")
plt.ylabel("n_estimators")
plt.title("train f1 score in heatmap")
plt.show()


# In[17]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_test_score")
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.ylabel("max_features")
plt.xlabel("n_estimators")
plt.title("train f1 score in heatmap")
plt.show()


# In[35]:


n_estimators = 10
max_features = .5


# In[44]:


model = RandomForestClassifier(max_depth = None, n_estimators = n_estimators, max_features = max_features, n_jobs = 3)
model.fit(x_train, y_train)


# In[36]:


y_train_pred = model.predict_proba(x_train)
y_test_pred = model.predict_proba(x_test)

# code taken from reference ipynb
pre_train, rec_train, tr_thresholds = precision_recall_curve(y_train, y_train_pred[:, 1])
pre_test, rec_test, te_thresholds = precision_recall_curve(y_test, y_test_pred[:, 1])

plt.plot(pre_train, rec_train, "ro-", label="train AUC =" + str(auc(rec_train, pre_train)))
plt.plot(pre_test, rec_test, "bo-", label="test AUC ="+ str(auc(rec_test, pre_test)))
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AU Precision-Recall plot for n_estimators = " + str(n_estimators) +  " max_features = " + str(max_features))
plt.grid()
plt.show()


# In[45]:


print("train f1 score:", f1_score(y_train, model.predict(x_train))*100)
print("test f1 score:", f1_score(y_test, model.predict(x_test))*100)


# # xgboost without SMOTE

# In[15]:


rf_model = xgb.XGBClassifier()
params = {"n_estimators": [10, 50, 100, 250],  
          "max_depth": [4, 8], 
          "subsample": [.1, .5, .9], 
          "colsample_bytree": [.1, .5, .9]
          }

randomsearch = RandomizedSearchCV(rf_model, params, cv = 3, n_jobs = 3, return_train_score = True, scoring = "f1", verbose = 10)
randomsearch.fit(x_train, y_train)


# In[19]:


results = pd.DataFrame.from_dict(randomsearch.cv_results_)
results


# In[21]:


plt.figure()
plt.plot(range(1, len(results["mean_test_score"])+1), results["mean_test_score"], "ro-", label = "mean_test_score")
plt.plot(range(1, len(results["mean_train_score"])+1), results["mean_train_score"], "bo-", label = "mean_train_score")
plt.legend()
plt.xlabel("indices")
plt.ylabel("f1 score")
plt.grid()
plt.show()


# In[40]:


n_estimators = 50
max_depth = 4
subsample = .9
colsample_bytree = .9


# In[41]:


model = xgb.XGBClassifier(max_depth = max_depth, n_estimators = n_estimators, subsample = subsample, n_jobs = 3, colsample_bytree = colsample_bytree)
model.fit(x_train, y_train)


# In[36]:


y_train_pred = model.predict_proba(x_train)
y_test_pred = model.predict_proba(x_test)

# code taken from reference ipynb
pre_train, rec_train, tr_thresholds = precision_recall_curve(y_train, y_train_pred[:, 1])
pre_test, rec_test, te_thresholds = precision_recall_curve(y_test, y_test_pred[:, 1])

plt.plot(pre_train, rec_train, "ro-", label="train AUC =" + str(auc(rec_train, pre_train)))
plt.plot(pre_test, rec_test, "bo-", label="test AUC ="+ str(auc(rec_test, pre_test)))
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AU Precision-Recall plot for n_estimators = " + str(n_estimators) +  " max_features = " + str(max_depth) + " subsample = " + str(subsample) + " colsample_bytree" + str(colsample_bytree))
plt.grid()
plt.show()


# In[43]:


print("train f1 score:", f1_score(y_train, model.predict(x_train))*100)
print("test f1 score:", f1_score(y_test, model.predict(x_test))*100)


# In[14]:


smote = SMOTE()
x_smote, y_smote = smote.fit_resample(x_train, y_train)
x_smote.shape


# # Random Forest with SMOTE

# In[15]:


rf_model = RandomForestClassifier(max_depth = None, n_jobs = -1)
params = {"n_estimators": [10, 50, 100, 250], 
          "max_features": [.1, .5, .9] 
          }

gridsearch = GridSearchCV(rf_model, params, cv = 3, n_jobs = 3, return_train_score = True, scoring = "f1", verbose = 10)
gridsearch.fit(x_smote, y_smote)


# In[16]:


results = pd.DataFrame.from_dict(gridsearch.cv_results_)
results.head()


# In[27]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_train_score")
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.xlabel("max_features")
plt.ylabel("n_estimators")
plt.title("train f1 score in heatmap")
plt.show()


# In[18]:


hmap = results.pivot("param_n_estimators", "param_max_features", "mean_test_score")
sns.heatmap(hmap, linewidth = 1, annot = True)
plt.ylabel("max_features")
plt.xlabel("n_estimators")
plt.title("train f1 score in heatmap")
plt.show()


# In[46]:


n_estimators = 10
max_features = .1


# In[47]:


model = RandomForestClassifier(max_depth = None, n_estimators = n_estimators, max_features = max_features, n_jobs = 3)
model.fit(x_smote, y_smote)


# In[25]:


y_train_pred = model.predict_proba(x_smote)
y_test_pred = model.predict_proba(x_test)

# code taken from reference ipynb
pre_train, rec_train, tr_thresholds = precision_recall_curve(y_smote, y_train_pred[:, 1])
pre_test, rec_test, te_thresholds = precision_recall_curve(y_test, y_test_pred[:, 1])

plt.plot(pre_train, rec_train, "ro-", label="train AUC =" + str(auc(rec_train, pre_train)))
plt.plot(pre_test, rec_test, "bo-", label="test AUC ="+ str(auc(rec_test, pre_test)))
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AU Precision-Recall plot for n_estimators = " + str(n_estimators) +  " max_features = " + str(max_features))
plt.grid()
plt.show()


# In[22]:


y_train_pred = model.predict_proba(x_smote)
y_test_pred = model.predict_proba(x_test)

# code taken from reference ipynb
pre_train, rec_train, tr_thresholds = precision_recall_curve(y_smote, y_train_pred[:, 1])
pre_test, rec_test, te_thresholds = precision_recall_curve(y_test, y_test_pred[:, 1])

plt.plot(pre_train, rec_train, "ro-", label="train AUC =" + str(auc(rec_train, pre_train)))
plt.plot(pre_test, rec_test, "bo-", label="test AUC ="+ str(auc(rec_test, pre_test)))
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AU Precision-Recall plot for n_estimators = " + str(n_estimators) +  " max_features = " + str(max_features))
plt.grid()
plt.show()


# In[48]:


print("train f1 score:", f1_score(y_train, model.predict(x_train))*100)
print("smote f1 score:", f1_score(y_smote, model.predict(x_smote))*100)
print("test f1 score:", f1_score(y_test, model.predict(x_test))*100)


# # xgboost with SMOTE

# In[28]:


rf_model = xgb.XGBClassifier()
params = {"n_estimators": [10, 50, 100, 250],  
          "max_depth": [4, 8], 
          "subsample": [.1, .5, .9], 
          "colsample_bytree": [.1, .5, .9]
          }

randomsearch = RandomizedSearchCV(rf_model, params, cv = 3, n_jobs = 3, return_train_score = True, scoring = "f1", verbose = 10)
randomsearch.fit(x_smote, y_smote)


# In[29]:


results = pd.DataFrame.from_dict(randomsearch.cv_results_)
results


# In[30]:


plt.figure()
plt.plot(range(1, len(results["mean_test_score"])+1), results["mean_test_score"], "ro-", label = "mean_test_score")
plt.plot(range(1, len(results["mean_train_score"])+1), results["mean_train_score"], "bo-", label = "mean_train_score")
plt.legend()
plt.xlabel("indices")
plt.ylabel("f1 score")
plt.grid()
plt.show()


# In[49]:


n_estimators = 100
max_depth = 4
subsample = .5
colsample_bytree = .5


# In[50]:


model = xgb.XGBClassifier(max_depth = max_depth, n_estimators = n_estimators, subsample = subsample, n_jobs = 3, colsample_bytree = colsample_bytree)
model.fit(x_smote, y_smote)


# In[33]:


y_train_pred = model.predict_proba(x_smote)
y_test_pred = model.predict_proba(x_test)

# code taken from reference ipynb
pre_train, rec_train, tr_thresholds = precision_recall_curve(y_smote, y_train_pred[:, 1])
pre_test, rec_test, te_thresholds = precision_recall_curve(y_test, y_test_pred[:, 1])

plt.plot(pre_train, rec_train, "ro-", label="train AUC =" + str(auc(rec_train, pre_train)))
plt.plot(pre_test, rec_test, "bo-", label="test AUC ="+ str(auc(rec_test, pre_test)))
plt.legend()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("AU Precision-Recall plot for n_estimators = " + str(n_estimators) +  " max_features = " + str(max_depth) + " subsample = " + str(subsample) + " colsample_bytree" + str(colsample_bytree))
plt.grid()
plt.show()


# In[51]:


print("train f1 score:", f1_score(y_train, model.predict(x_train))*100)
print("smote f1 score:", f1_score(y_smote, model.predict(x_smote))*100)
print("test f1 score:", f1_score(y_test, model.predict(x_test))*100)


# In[ ]:




