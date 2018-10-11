
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_table('MULTIVAR_FIG5-2.DAT')


# In[4]:


df.index = df.index + 1


# In[5]:


df.head()


# <br>

# # <font size=12>#1</font>

# ## <center><b>1. Make 3 plots and examine the correlation among the variables </b></center>

# In[6]:


sns.lmplot(x='X2', y='X1', data=df)
plt.title('X1 vs X2', fontsize=18)

sns.lmplot(x='X3', y='X2', data=df)
plt.title('X2 vs X3', fontsize=18)

sns.lmplot(x='X1', y='X3', data=df)
plt.title('X3 vs X1', fontsize=18)


# In[7]:


df.corr()


# <br>

# ## <center><b>2. Compute the normalized variables (Z1,Z2,Z3) corresponding to X1,X2,X3 respectively </b></center>

# In[8]:


mu1 = np.mean(df['X1'])
mu2 = np.mean(df['X2'])
mu3 = np.mean(df['X3'])


# In[9]:


var1 = np.var(df['X1'])
var2 = np.var(df['X2'])
var3 = np.var(df['X3'])


# In[10]:


z1 = pd.Series(df['X1'].apply(lambda x: (x-mu1)/var1), name='z1')
z2 = pd.Series(df['X2'].apply(lambda x: (x-mu2)/var2), name='z2')
z3 = pd.Series(df['X3'].apply(lambda x: (x-mu3)/var3), name='z3')


# In[11]:


"""concat joins the 3 series together to make the Z table, and as_matrix
converts to a 2D array for matrix operations"""

Z = pd.concat([z1,z2,z3], axis=1).as_matrix()


# In[12]:


print('Z')
print(Z)


# <br>

# ## <center><b>3. Compute covariance matrix   $C=\frac{Z^{T}Z}{(n-1)}$ </b></center>

# In[13]:


Z_T = np.transpose(Z)


# In[14]:


C = np.matmul(Z_T,Z) / (len(Z)-1)


# In[15]:


print('C')
print(C)


# <br>

# In[16]:


## numpy also has a built in function to get the same result


# In[17]:


np.cov(Z, rowvar=False)


# <br>

# ## <center><b>4. Perform SVD of the covariance matrix and compute the principle components.<br><br>How many are necessary to preserve at least 90% of the variance in the original data?</b></center>

# In[18]:


U, s, vh = np.linalg.svd(C)


# <br>

# In[19]:


print('U')
print(U)


# <br>

# In[20]:


print('vh')
print(vh)


# <br>

# In[21]:


print('s')
print(s)


# <br>

# In[22]:


eigval,eigvector = np.linalg.eig(C)
print('Eigenvalues:', eigval)
print('\b')
print('Eigenvectors:')
print(eigvector)


# <br>

# In[23]:


total = np.sum(s)


# In[24]:


print('\b')
print('PC1 represents', round(s[0]/total*100,2),'%')
print('\b')
print('PC2 represents', round((s[0]+s[1])/total*100,2),'%')
print('\b')


# <br>

# ## <center><b>5.  Plot PC2 versus PC1 and compute the variances along each axis</b></center>

# In[25]:


V1 = eigvector[:,0].reshape(3,1)
V2 = eigvector[:,1].reshape(3,1)


# In[26]:


PC1 = np.matmul(Z,V1)
PC2 = np.matmul(Z,V2)
PrinComps = pd.DataFrame(np.concatenate([PC1,PC2], axis=1), columns=['PC1','PC2'])


# In[27]:


PrinComps


# In[28]:


fig,axes = plt.subplots(figsize=(10,7))
axes.set_xlabel('PC1',fontsize=18)
axes.set_ylabel('PC2', fontsize=18)
plt.scatter(x=PC1, y=PC2)


# In[29]:


sns.lmplot(x='PC1', y='PC2', data=PrinComps, height=8)


# In[30]:


print('The variance along PC1 axis is',np.var(PC1))
print('\b')
print('The variance along PC2 axis is',np.var(PC2))


# <br>

# # <font size=12>#2</font>

# ## <center><b>Using the principle components, perform K-mean clustering and divide the data </b></center>

# In[31]:


from sklearn.cluster import KMeans


# In[32]:


kmeans = KMeans(n_clusters=2)


# In[33]:


kmeans.fit(PrinComps)


# In[34]:


kmeans.labels_


# In[35]:


PrinComps['Cluster'] = pd.Series(kmeans.labels_)


# In[36]:


PrinComps.head()


# In[37]:


sns.lmplot(x='PC1', y='PC2', data=PrinComps, height=8,hue='Cluster', fit_reg=False, scatter_kws={"s": 150})

