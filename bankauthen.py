#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('bank_note_data.csv')
print(data.head())
print(data.describe())


# In[2]:


sns.set_style('darkgrid')
sns.countplot(x='Class',data=data)
plt.show()


# In[3]:

sns.pairplot(data,vars=data.columns[:-1], hue='Class', palette="GnBu_d")
plt.show()


# In[4]:


sns.pairplot(data,vars=data.columns[:-1], hue='Class', palette="GnBu_d")
plt.show()


# In[5]:


scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
print(df_feat.head())


# In[6]:


X = df_feat
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(df_feat.columns)


# In[7]:


image_var = tf.compat.v1.feature_column.numeric_column("Image.Var")
image_skew = tf.compat.v1.feature_column.numeric_column('Image.Skew')
image_curt = tf.compat.v1.feature_column.numeric_column('Image.Curt')
entropy = tf.compat.v1.feature_column.numeric_column('Entropy')
feat_cols = [image_var,image_skew,image_curt,entropy]


# In[8]:


classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)


# In[9]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)


# In[10]:


classifier.train(input_fn = input_func,steps = 500)


# In[11]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[12]:


note_predictions = list(classifier.predict(input_fn=pred_fn))


# In[13]:


note_predictions[0]


# In[14]:


final_preds = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])


# In[15]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[16]:


print(confusion_matrix(y_test,final_preds))


# In[17]:


print(classification_report(y_test,final_preds))
print(accuracy_score(y_test,final_preds))


# In[18]:


from sklearn.ensemble import RandomForestClassifier


# In[19]:


rfc = RandomForestClassifier(n_estimators=200)


# In[20]:


rfc.fit(X_train,y_train)


# In[21]:


rfc_preds = rfc.predict(X_test)


# In[22]:


print(classification_report(y_test,rfc_preds))


# In[23]:


print(confusion_matrix(y_test,rfc_preds))


# In[ ]:

print(accuracy_score(y_test,rfc_preds))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




