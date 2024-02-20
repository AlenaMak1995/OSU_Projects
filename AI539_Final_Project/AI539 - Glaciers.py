#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Missing-values" data-toc-modified-id="Missing-values-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Missing values</a></span><ul class="toc-item"><li><span><a href="#Remove-items" data-toc-modified-id="Remove-items-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Remove items</a></span></li></ul></li><li><span><a href="#Impute-with-average" data-toc-modified-id="Impute-with-average-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Impute with average</a></span></li><li><span><a href="#Impute-with-Linear-Regression" data-toc-modified-id="Impute-with-Linear-Regression-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Impute with Linear Regression</a></span></li><li><span><a href="#Outliers" data-toc-modified-id="Outliers-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Outliers</a></span><ul class="toc-item"><li><span><a href="#Do-nothing" data-toc-modified-id="Do-nothing-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Do nothing</a></span></li><li><span><a href="#IQR" data-toc-modified-id="IQR-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>IQR</a></span></li><li><span><a href="#Square-Root-Transformation" data-toc-modified-id="Square-Root-Transformation-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Square Root Transformation</a></span></li></ul></li><li><span><a href="#Disbalance" data-toc-modified-id="Disbalance-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Disbalance</a></span><ul class="toc-item"><li><span><a href="#Balance-=-'weighted'" data-toc-modified-id="Balance-=-'weighted'-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Balance = 'weighted'</a></span></li><li><span><a href="#Upsampling" data-toc-modified-id="Upsampling-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Upsampling</a></span></li><li><span><a href="#Downsampling" data-toc-modified-id="Downsampling-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Downsampling</a></span></li></ul></li></ul></div>

# In[1]:


print('AI539-Machine Learning Challenges, Final Project')
print('Alena Makarova, 03/20/2023')


# In[2]:


# !pip install dataframe_image
# import dataframe_image as dfi
# !pip install folium
#!pip install missingno
# !pip install imbalanced-learn
# # check version number
# import imblearn
# print(imblearn.__version__)


# In[3]:





# In[4]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import astropy
import seaborn as sns
import pylab as pl
from sklearn.tree import DecisionTreeClassifier

from astropy.visualization import hist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import folium
import missingno as msno 

# Import methods for hyperparameter's tunning and accuracy_score

from sklearn.metrics import accuracy_score
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from sklearn.linear_model import LinearRegression
from sklearn import linear_model

import plotly.express as px

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
from sklearn.metrics import confusion_matrix, classification_report


from collections import Counter
from matplotlib import pyplot

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss
import imblearn


# In[5]:


#Read the file with try-exept method
try:
    df = pd.read_csv('new_set.csv', low_memory=False)
except:
    print('The required file does not exist!')    
#Read first 5 lines of the file
#display(df.head())


# In[6]:


df.info()


# In[7]:


df = df.rename(columns = {' glacier_name':'glacier_name', ' lat':'lat', 
                          ' lon':'lon', ' total_area':'total_area', 
                         ' max_length':'max_length', ' max_elev':'max_elev',' min_elev':'min_elev',
                          ' topo_year':'topo_year', ' primary_class':'primary_class',
                         ' tongue_activity ': 'tongue_activity', ' form':'form', ' frontal_char':'frontal_char',
                         ' longi_profile':'longi_profile', ' source_nourish':'source_nourish'})


# In[8]:


# Создаем функцию перевода значения колонок в числовой тип 
def convert_to_numeric (value):
    df[value] = pd.to_numeric(df[value], errors = 'coerce')
#  Чтобы не переписывать вручную все названия колонок, выделим названия столбцов
# фрейма в list и используем метод try-except, так как не все данные числового типа.
list_of_cols = df.columns.tolist()
# for value in df.columns.tolist():
#     try:
#         convert_to_numeric(value)
#     except:
#         'None'
del list_of_cols[8]   


# In[9]:


for value in list_of_cols:
    try:
        convert_to_numeric(value)
    except:
        'None'


# In[10]:


df['topo_year'] = pd.to_datetime(df['topo_year'], errors = 'coerce')


# In[11]:


df.info()
#dfi.export(df.info(),"df_info().png")


# In[12]:


#let's draw a plot of missing values
df = df.drop(['wgi_glacier_id', 'glacier_name'], axis = 1)
missing_values = df.isna().sum()
missing_values = missing_values.to_frame()
missing_values = missing_values.sort_values(by=[0], ascending = False)
missing_values = missing_values.rename(columns = {0:'values'})
#missing_values = missing_values.drop(['wgi_glacier_id', 'glacier_name'], axis = 1)
missing_values.plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.title('The number of missing values in features')
plt.xlabel('Columns in data set')
plt.ylabel('Number of missing values')
#dfi.export(missing_values,"missing.png")
plt.autoscale()
#plt.savefig('new_values.png', bbox_inches = 'tight')


# In[13]:


#find the duplicates
print('# of diuplictes', df.duplicated().sum())
#delete the duplicates
df = df.drop_duplicates()
print('# of suplicates', df.duplicated().sum())


# In[14]:


#delete the duplicates
df = df.drop_duplicates()


# In[15]:


results = []
def append_results(feature, type_, min_,max_, mean_, median_ ):
    a = results.append({'Feature': feature, 'Type':type_, 'Min':min_, 'Max':max_, 'Mean':mean_, 'Median':median_}) 


# In[16]:


def plot_hist(df, value, number_of_bins, first_boarder, last_boarder, title, feature, result ):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title(feature)
    ax[1].set_title(title)
    sns.violinplot(x = value, data=df, palette="Set3", cut=2, linewidth=3, inner="box", ax=ax[0])
    #plt.savefig('sns'+str(value)+'.png', bbox_inches = 'tight')
    plt.xlabel(result,size = 10,alpha=1)
    graph = df[value].hist(range = (first_boarder, last_boarder),bins = number_of_bins, ax=ax[1])
    #plt.xlabel(feature,size = 16,alpha=0.7)
    
    plt.title(title)
    plt.show()
    append_results(value, df[value].dtypes, np.nanmin(df[value]), np.nanmax(df[value]),(df[value]).mean(), (df[value]).median())
    #plt.savefig('hist'+str(value)+'.png', bbox_inches = 'tight')
    return df[value].describe()#, graph


# In[17]:


plot_hist(df, 'lat', 100, -46, 82, 'Latitudes of glaciers', 'Latitude', 'Decimal degrees')
#plt.savefig('lat.png', bbox_inches = 'tight')


# In[18]:


plot_hist(df, 'lon', 100, -142, 176, 'Longitudes of glaciers', 'Longitude', 'Decimal degrees')
#plt.savefig('lon.png', bbox_inches = 'tight')


# In[19]:


plot_hist(df, 'total_area', 100, 0, 1, 'The total area of the glacier', 'Total area', 'Square kilometers')
#plt.savefig('area.png', bbox_inches = 'tight')


# In[20]:


plot_hist(df, 'max_elev', 100, 0, 6900, 'The max elevation of the glacier', 'Max elevation', 'Meters above sea level')
#plt.savefig('max_elev.png', bbox_inches = 'tight')


# In[21]:


plot_hist(df, 'min_elev', 100, 0, 5700, 'The min elevation of the glacier', 'Min elevation', 'Meters above sea level')
#plt.savefig('min_elev.png', bbox_inches = 'tight')


# In[22]:


plot_hist(df, 'primary_class', 10, 0, 9, 'The primary classification of the glacier', 'Primary classification', 'Code') 
#plt.savefig('primary.png', bbox_inches = 'tight')


# In[23]:


plot_hist(df, 'tongue_activity', 9, 1, 8, 'The activity of the tongue of the glacier', 'Tongue activity', 'Code') 
#plt.savefig('tongue.png', bbox_inches = 'tight')


# In[24]:


plot_hist(df, 'max_length', 100, 0, 10, 'The max length of the glacier', 'Max length', 'Kilometers')
#plt.savefig('max_length.png', bbox_inches = 'tight')


# In[25]:


plot_hist(df, 'form', 10, 0, 9, 'The form of the glacier', 'Form codes', 'Code')
#plt.savefig('form.png')


# In[26]:


plot_hist(df, 'frontal_char', 10, 0, 9, 'The frontal characteristics', 'Frontal characteristics', 'Code')
#plt.savefig('frontchar.png', bbox_inches = 'tight')


# In[27]:


plot_hist(df, 'source_nourish', 4, 0, 3, 'The source of nourishment', 'Source', 'Code')
#plt.savefig('source.png', bbox_inches = 'tight')


# In[28]:


results_df = pd.DataFrame(results)
#dfi.export(results_df,"describe_values.png")
#results_df


# In[29]:


held_out = df[df['topo_year'] > '1975-01-01']
df = df[df['topo_year'] <= '1975-01-01']


# In[30]:


#let's plot the map with all glaciers
world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')
purpose_colour = {1.0:'yellow', 2.0:'gold', 3.0: 'green', 4.0:'cyan', 5.0 : 'blue', 6.0:'purple', 7.0:'orchid', 8.0:'red'}
map = folium.Map(location=[df.lat.mean(), df.lon.mean()], 
                 zoom_start=3, control_scale=True)
for i,row in df.iterrows():
    #Setup the content of the popup
    iframe = folium.IFrame(f'Well Name: {str(row["total_area"])} \n Purpose: {str(row["tongue_activity"])}')
    
    #Initialise the popup using the iframe
    popup = folium.Popup(iframe, min_width=150, max_width=150)
    
    try:
        icon_color = purpose_colour[row['tongue_activity']]
    except:
        #Catch nans
        icon_color = 'gray'
    
    #Add each row to the map
    folium.CircleMarker(location=[row['lat'],row['lon']],
                        popup = popup, color=icon_color).add_to(map)


# In[31]:


#print(map)


# In[32]:


matrix = pd.plotting.scatter_matrix(df, figsize=(20, 20), alpha = 0.03, diagonal='kde')
#plt.savefig('matrix.png', bbox_inches = 'tight')


# In[33]:


#let's look at the missing values
sorted = df.sort_values('total_area')
msno.matrix(sorted)
plt.savefig('missing.png', bbox_inches = 'tight')
#msno.matrix(df)


# In[34]:


# heatmap for missing values
msno.heatmap(df, cmap='YlGnBu')
#plt.savefig('heat.png', bbox_inches = 'tight')


# # Missing values

# In[35]:


# let's make a table for all results 
table_1 = []
table_2 = []
table_3 = []
table_4 = []
table_5 = []
table_6 = []

def append_results_1(model, acc_1, acc_2, acc_3):
    a = table_1.append({'Model': model, 'Remove items': acc_1, 'Impute values': acc_2, 'LRImputation': acc_3})

def append_results_2(model, f1_1, f1_2, f1_3):
     b = table_2.append({'Model': model, 'Remove items': f1_1, 'Impute values': f1_2, 'LRImputation': f1_3})
    
def append_results_3(model, acc_1, acc_2, acc_3):
    c = table_3.append({'Model': model, 'Do nothing': acc_1, 'IQR': acc_2, 'Square Root Transformation': acc_3})
    
def append_results_4(model, f1_1, f1_2, f1_3):
    c = table_4.append({'Model': model, 'Do nothing': f1_1, 'IQR': f1_2, 'Square Root Transformation': f1_3})

def append_results_5(model, acc_1, acc_2, acc_3):
    c = table_5.append({'Model': model, 'Balance weight': acc_1, 'Upsampling': acc_2, 'Downsampling': acc_3})
    
def append_results_6(model, f1_1, f1_2, f1_3):
    c = table_6.append({'Model': model, 'Balance weight': f1_1, 'Upsampling': f1_2, 'Downsampling': f1_3})
  


# ## Remove items

# In[36]:


#delete the feature topo_year as we wouldn't use it in the training process
df = df.drop(['topo_year'], axis = 1)
held_out = held_out.drop(['topo_year'], axis = 1)


# In[37]:


def preprocess_train(df1):
    # create the features
    df1['longi_profile'] = df1['longi_profile'].astype('int')
    df1['source_nourish'] = df1['source_nourish'].astype('int')
    df1['tongue_activity'] = df1['tongue_activity'].astype('int')

    
    features = df1.drop([ 'tongue_activity'], axis=1)
    # and target
    target = df1['tongue_activity']
    print('Train features shape: ', features.shape)
    target = target.astype('int')
    # Create features_train and features_val using train_test_split method
    features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=12345)
    print('Train features shape: ', features_train.shape)
    print('Val features shape: ', features_val.shape)
    
    return features_train, features_val, target_train, target_val


# In[38]:


def preprocess_test(df2):
    # create the features
    
    df2['longi_profile'] = df2['longi_profile'].astype('int')
    df2['source_nourish'] = df2['source_nourish'].astype('int')
    df2['tongue_activity'] = df2['tongue_activity'].astype('int')
    
    # create the features
    features_test = df2.drop(['tongue_activity'], axis=1)
    # and target
    target_test = df2['tongue_activity']
    print('Test features shape: ', features_test.shape)
    target_test = target_test.astype('int')
    
    return features_test, target_test


# In[39]:


# # function to scale the features
# def scaler(features_fit, features_transform):
#     #declare columns that we want to scale
#     numeric = ['lat', 'lon', 'min_elev', 'max_elev', 'total_area', 'max_length']
#     #scale the data
#     scaler = StandardScaler()
#     # fit on the training set
#     scaler.fit(features_fit[numeric])
#     # scale train and val sets
#     features_transform[numeric] = scaler.transform(features_transform[numeric])
#     return features_transform


# In[40]:


# delete missing values in training set
df_misval_one = df.dropna()
#test_misval_one = held_out.dropna()


# In[41]:


df_misval_one.info()


# In[42]:


features_train_1, features_val_1, target_train_1, target_val_1 = preprocess_train(df_misval_one)


# In[43]:


# scaler(features_train_1, features_train_1)
# scaler(features_train_1, features_val_1)


# In[44]:


# heatmap for features correlation
plt.figure(figsize = (10,10))
#features_df = features_train.drop(['max_elev', 'min_elev'],  axis=1)
sns.heatmap(features_train_1.corr(), cmap ='RdYlGn', linewidths = 0.30, annot = True)
#plt.savefig('heatmap.png')


# In[45]:


# let's write a function for testing a model
def test_model(model, features_test,target_test ):
    predictions = model.predict(features_test)
    result = accuracy_score(target_test, predictions)
    result_f1 = f1_score(target_test, predictions, average='macro') 
    print("Best test result: ",result, "Best F1 test result: ", result_f1 )
    return result, result_f1


# In[46]:


test_misval_one= held_out.copy()


# In[47]:


#let's impute the missing values in a held-out data set
test_misval_one.isna().sum()


# In[48]:


# let's impute the missing values in test data set with average from the training data set
for col in list(test_misval_one.columns):
    test_misval_one[col] = test_misval_one[col].fillna(df_misval_one[col].mean())


# In[49]:


features_test_1, target_test_1 = preprocess_test(test_misval_one)
#scaler(features_train_1, features_test_1)


# In[50]:


# let's train DummyClassifier to get baseline accuracy
def DClf(features_train, features_val, target_train, target_val, strategy):
    # assign the model
    dummy_clf = DummyClassifier(strategy=strategy, random_state = 12345)
    # fitting the model
    dummy_clf.fit(features_train, target_train)
    # make predvvictions
    prediction = dummy_clf.predict(features_val)
    dummy_result_acc = accuracy_score(target_val, prediction)
    dummy_result_f1 = f1_score(target_val, prediction, average='macro') 
    print('Baseline accuracy: ', dummy_result_acc, 'F1-score for baseline:', dummy_result_f1)
    return dummy_result_acc, dummy_result_f1, dummy_clf


# In[51]:


# train dummy with 'most_frequent' strategy
acc_mf_train_1, f1_mf_train_1, model_train_mf_1 = DClf(features_train_1, features_val_1, target_train_1, target_val_1, 'most_frequent')
# test dummy with 'most_frequent' strategy
acc_mf_test_1, f1_mf_test_1 = test_model(model_train_mf_1, features_test_1, target_test_1)
# train dummy with 'stratified' strategy
acc_s_train_1, f1_s_train_1, model_train_s_1 = DClf(features_train_1, features_val_1, target_train_1, target_val_1,'stratified')
acc_s_test_1, f1_s_test_1 = test_model(model_train_s_1, features_test_1, target_test_1)


# In[52]:


#declare the function for RandomForestClassifier with depth and est in range 1:25
def RFC(features_train, features_val, target_train, target_val, Balance = None):
    # let's train RandomForestClassifier
    best_model = None
    best_result_forest = 0
    best_forest_depth = 0
    best_est = 0
    # first, let's find the best depth
    for depth in range(1, 25, 1):  
        for est in range (1, 25, 1):   
            # assign the model
            model_forest = RandomForestClassifier(random_state = 12345, max_depth = depth,n_estimators = est, criterion='gini', class_weight = Balance)
            # fitting the model
            model_forest.fit(features_train, target_train)       
            # make predictions
            predictions_test_forest = model_forest.predict(features_val)
            result = accuracy_score(target_val, predictions_test_forest)
            result_f1 = f1_score(target_val, predictions_test_forest, average='macro') 
            print('Accuracy: ', result, 'f1: ',result_f1, 'depth: ', depth, 'est', est)
            # let's find the best value and the best depth of our model
            if result > best_result_forest:
                best_result_forest = result
                best_forest_depth = depth
                best_model = model_forest
                best_est = est
    print('Best result:', best_result_forest, 'Best depth: ',best_forest_depth, "Best # of trees: ", best_est, 'Best F1:',result_f1 )            
    return best_result_forest, result_f1, best_model


# In[53]:


best_result_forest_1, result_f1_1, best_model_1 = RFC(features_train_1, features_val_1, target_train_1, target_val_1)


# In[54]:


acc_forest_test_1, f1_forest_test_1 = test_model(best_model_1, features_test_1, target_test_1)


# In[55]:


#declare the function for DecisionTreeClassifier with depth 1:20
def DTC(features_train, features_val, target_train, target_val, Balance = None):# let's train DecissionTreeClassifier
    best_result = 0
    best_depth = 0
    best_model = None
    # iterate through the values of depth
    for depth in range(1, 20, 1):  
        # assign the model
        model_tree = DecisionTreeClassifier(random_state = 12345, max_depth = depth, criterion='gini', class_weight = Balance)
        # fitting the model
        model_tree.fit(features_train, target_train)
        # make predictions
        predictions_test_tree = model_tree.predict(features_val)
        result = accuracy_score(target_val, predictions_test_tree) 
        result_f1 = f1_score(target_val, predictions_test_tree, average='macro') 
        #result_f1 = f1_score(target_val, predictions_test_tree, average='macro') 
        print('Accuracy: ', result,  'depth: ', depth)
        #let's find the best value and the best depth of our model
        if result > best_result:
            best_result = result
            best_depth = depth
            best_model = model_tree
    print('The best accuracy of the DecisionTreeClassifier: ', best_result, "Best depth: ", best_depth, 'Best F1:', result_f1)
    return best_result, result_f1, best_model


# In[56]:


best_result_tree_1, result_tree_f1_1, best_model_tree_1 = DTC(features_train_1, features_val_1, target_train_1, target_val_1)


# In[57]:


acc_tree_test_1, f1_tree_test_1 = test_model(best_model_tree_1, features_test_1, target_test_1)


# In[58]:


def KNN(features_train, features_val, target_train, target_val): 
    # assign the model
    model_knn = KNeighborsClassifier(n_neighbors=9)
    # fitting the model
    model_knn.fit(features_train, target_train)
    # make predictions
    predictions_test_knn = model_knn.predict(features_val)
    # find the accuracy
    results_knn = accuracy_score(target_val, predictions_test_knn) 
    result_f1 = f1_score(target_val, predictions_test_knn, average='macro') 
    #result_f1 = f1_score(target_val, regression_prediction, average='macro') 
    print('The best accuracy of the 9-KNeighborsClassifier: ', results_knn, 'Best F1:', result_f1)  
    return results_knn, result_f1, model_knn


# In[59]:


best_result_KNN_1, result_KNN_f1_1, best_model_KNN_1 = KNN(features_train_1, features_val_1, target_train_1, target_val_1)


# In[60]:


acc_KNN_test_1, f1_KNN_test_1 = test_model(best_model_KNN_1, features_test_1, target_test_1)


# # Impute with average

# In[61]:


impute_df = df.copy()
print('Train data set before imputation:\n', impute_df.isna().sum())
impute_test = held_out.copy()
print('Test data set before imputation:\n', impute_test.isna().sum())

list_of_cols = df.columns.tolist()
for val in list_of_cols:
    impute_df[val].fillna(value=int(impute_df[val].mean()), inplace=True)
    impute_test[val].fillna(value=int(impute_test[val].mean()), inplace=True)
    


# In[62]:


features_train_2, features_val_2, target_train_2, target_val_2 = preprocess_train(impute_df)
# scaler(features_train_2, features_train_2)
# scaler(features_train_2, features_val_2)


# In[63]:


best_result_forest_2, result_f1_2, best_model_2 = RFC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[64]:


acc_forest_test_2, f1_forest_test_2 = test_model(best_model_2, features_test_1, target_test_1)


# In[65]:


best_result_tree_2, result_f1_tree_2, best_model_tree_2 = DTC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[66]:


acc_tree_test_2, f1_tree_test_2 = test_model(best_model_tree_2, features_test_1, target_test_1)


# In[67]:


best_result_KNN_2, result_f1_KNN_2, best_model_KNN_2 = KNN(features_train_2, features_val_2, target_train_2, target_val_2)


# In[68]:


acc_KNN_test_2, f1_KNN_test_2 = test_model(best_model_KNN_2, features_test_1, target_test_1)


# # Impute with Linear Regression

# In[69]:


def random_imputation(df, feature):
    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    return df


# In[70]:


#let's make a function to impute missing values with Linear Regression
#https://www.kaggle.com/code/shashankasubrahmanya/missing-data-imputation-using-regression/notebook
def impute_LR(df):
    new_df = df.copy()
    #new_df = new_df.drop(['topo_year'], axis = 1)
    print(new_df.isna().sum())
    
    missing_columns = []
    for feature in new_df.columns.tolist():
        if new_df[feature].isnull().sum() > 0:
            missing_columns.append(feature)
    
    for feature in missing_columns:
        new_df[feature + '_imp'] = new_df[feature]
        new_df = random_imputation(new_df, feature)      
      
    deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

    for feature in missing_columns:
        
        deter_data["Det" + feature] = new_df[feature + '_imp']
        parameters = list(set(new_df.columns) - set(missing_columns) - {feature + '_imp'})
    
        #Create a Linear Regression model to estimate the missing data
        model = linear_model.LinearRegression()
        model.fit(X = new_df[parameters], y = new_df[feature + '_imp'])
    
        #observe that I preserve the index of the missing data from the original dataframe
        new_df.loc[new_df[feature].isnull(), feature] = model.predict(new_df[parameters])[new_df[feature].isnull()] 
        
    for col in missing_columns:
        new_df = new_df.drop(col + '_imp', axis = 1)      
    return new_df      


# In[71]:


linear_df = impute_LR(df)


# In[72]:


linear_df['tongue_activity'] = linear_df['tongue_activity'].astype('int')


# In[73]:


# class_frequency = linear_df['tongue_activity'].value_counts(normalize=True)
# class_frequency


# In[74]:


features_train_3, features_val_3, target_train_3, target_val_3= preprocess_train(linear_df)
# scaler(features_train_3, features_train_3)
# scaler(features_train_3, features_val_3)


# In[75]:


best_result_forest_3, result_f1_3, best_model_3 = RFC(features_train_3, features_val_3, target_train_3, target_val_3)


# In[76]:


acc_forest_test_3, f1_forest_test_3 = test_model(best_model_3, features_test_1, target_test_1)


# In[77]:


best_result_tree_3, result_f1_tree_3, best_model_tree_3 = DTC(features_train_3, features_val_3, target_train_3, target_val_3)


# In[78]:


acc_tree_test_3, f1_tree_test_3 = test_model(best_model_tree_3, features_test_1, target_test_1)


# In[79]:


best_result_KNN_3, result_f1_KNN_3, best_model_KNN_3 = KNN(features_train_3, features_val_3, target_train_3, target_val_3)


# In[80]:


acc_KNN_test_3, f1_KNN_test_3 = test_model(best_model_KNN_3, features_test_1, target_test_1)


# In[81]:


table_1 = []
table_2 = []
table_3 = []
table_4 = []
table_exp = []


# In[82]:


append_results_1('DecisionTreeClassifier', acc_tree_test_1, acc_tree_test_2, acc_tree_test_3)
append_results_1('RandomForestClassifier', acc_forest_test_1, acc_forest_test_2, acc_forest_test_3)
append_results_1('9-Nearest Neighbors', acc_KNN_test_1, acc_KNN_test_2, acc_KNN_test_3)
append_results_1('DummyClassifier (most_frequent)', acc_mf_test_1, acc_mf_test_1, acc_mf_test_1)
append_results_1('DummyClassifier (stratified)', acc_s_test_1, acc_s_test_1, acc_s_test_1)


# In[83]:


first = pd.DataFrame(table_1)
first


# In[84]:


first.loc[:,'Remove items':] = (first.loc[:,'Remove items':] * 100).round(2) 
first
#dfi.export(first,"first.png")


# In[85]:


append_results_2('DecisionTreeClassifier', f1_tree_test_1, f1_tree_test_2, f1_tree_test_3)
append_results_2('RandomForestClassifier', f1_forest_test_1, f1_forest_test_2, f1_forest_test_3)
append_results_2('9-Nearest Neighbors', f1_KNN_test_1, f1_KNN_test_2, f1_KNN_test_3)
append_results_2('DummyClassifier (most_frequent)', f1_mf_test_1, f1_mf_test_1, f1_mf_test_1)
append_results_2('DummyClassifier (stratified)', f1_s_test_1, f1_s_test_1, f1_s_test_1)


# In[86]:


second = pd.DataFrame(table_2)
second


# In[87]:


second.loc[:,'Remove items':] = (second.loc[:,'Remove items':] * 100).round(2) 
second
#dfi.export(second,"second.png")


# # Outliers

# ## Do nothing 

# ## IQR

# In[88]:


def IQR(col, df):
    # Найдем стандартное отклонение
    std_df = df[col].std()
    # среднее
    mean_df = df[col].mean()
    # ограничим верхнюю границу значением суммы среднего и трех стандартных отклонений
    upper_bound = mean_df + 3*std_df
    lower_bound = mean_df - 3*std_df
    iqr_df = df[df[col] < upper_bound]
    iqr_df = iqr_df[iqr_df[col] > lower_bound]
    iqr_df = iqr_df.reset_index(drop = True)
    return iqr_df


# In[89]:


iqr_df = impute_df.copy()


# In[90]:


plot_hist(iqr_df, 'total_area', 100, 0, 4, 'The total area of the glacier', 'Total area', 'Square kilometers')


# In[91]:


iqr_df = IQR('total_area', iqr_df)


# In[92]:


plot_hist(iqr_df, 'total_area', 100, 0, 4, 'The total area of the glacier', 'Total area', 'Square kilometers')


# In[93]:


plot_hist(iqr_df, 'max_length',100, 1, 19, 'The max length of the glacier new', 'Max length', 'Length')


# In[94]:


iqr_df = IQR('max_length', iqr_df)


# In[95]:


plot_hist(iqr_df, 'max_length',100, 1, 19, 'The max length of the glacier new', 'Max length', 'Length')


# In[96]:


plot_hist(iqr_df, 'max_elev', 100, 0, 6900, 'The max elevation of the glacier', 'Max elevation', 'Meters above sea level')


# In[97]:


iqr_df = IQR('max_elev', iqr_df)


# In[98]:


plot_hist(iqr_df, 'max_elev', 100, 0, 6900, 'The max elevation of the glacier', 'Max elevation', 'Meters above sea level')


# In[99]:


plot_hist(iqr_df, 'min_elev', 100, 0, 5900, 'The min elevation of the glacier', 'Min elevation', 'Meters above sea level') 


# In[100]:


iqr_df = IQR('min_elev', iqr_df)


# In[101]:


plot_hist(iqr_df, 'min_elev', 100, 0, 5700, 'The min elevation of the glacier', 'Min elevation', 'Meters above sea level') 


# In[102]:


#preprocess the features and target for IQR strategy
features_train_4, features_val_4, target_train_4, target_val_4 = preprocess_train(iqr_df)


# In[103]:


#### train dummy with 'most_frequent' strategy
acc_mf_train_2, f1_mf_train_2, model_train_mf_2 = DClf(features_train_4, features_val_4, target_train_4, target_val_4, 'most_frequent')
# test dummy with 'most_frequent' strategy
acc_mf_test_2, f1_mf_test_2 = test_model(model_train_mf_2, features_test_1, target_test_1)
# train dummy with 'stratified' strategy
acc_s_train_2, f1_s_train_2, model_train_s_2 = DClf(features_train_4, features_val_4, target_train_4, target_val_4,'stratified')
acc_s_test_2, f1_s_test_2 = test_model(model_train_s_2, features_test_1, target_test_1)


# In[104]:


best_result_forest_4, result_f1_4, best_model_4 = RFC(features_train_4, features_val_4, target_train_4, target_val_4)


# In[105]:


acc_forest_test_4, f1_forest_test_4 = test_model(best_model_4, features_test_1, target_test_1)


# In[106]:


best_result_tree_4, result_f1_tree_4, best_model_tree_4 = DTC(features_train_4, features_val_4, target_train_4, target_val_4)


# In[107]:


acc_tree_test_4, f1_tree_test_4 = test_model(best_model_tree_4, features_test_1, target_test_1)


# In[108]:


best_result_KNN_4, result_f1_KNN_4, best_model_KNN_4 = KNN(features_train_4, features_val_4, target_train_4, target_val_4)


# In[109]:


acc_KNN_test_4, f1_KNN_test_4 = test_model(best_model_KNN_4, features_test_1, target_test_1)


# ## Square Root Transformation

# In[110]:


# let's make a copy of our dataframe
sqrt_df = impute_df.copy()


# In[111]:


# take a square root of the values in our training set
sqrt_df['total_area'] = np.sqrt(sqrt_df['total_area'])
sqrt_df['max_length'] = np.sqrt(sqrt_df['max_length'])
sqrt_df['max_elev'] = np.sqrt(sqrt_df['max_elev'])
sqrt_df['min_elev'] = np.sqrt(sqrt_df['min_elev'])


# In[112]:


# train the forest
features_train_5, features_val_5, target_train_5, target_val_5 = preprocess_train(sqrt_df)


# In[113]:


best_result_forest_5, result_f1_5, best_model_5 = RFC(features_train_5, features_val_5, target_train_5, target_val_5)


# In[114]:


acc_forest_test_5, f1_forest_test_5 = test_model(best_model_5, features_test_1, target_test_1)


# In[115]:


best_result_tree_5, result_f1_tree_5, best_model_tree_5 = DTC(features_train_5, features_val_5, target_train_5, target_val_5)


# In[116]:


acc_tree_test_5, f1_tree_test_5 = test_model(best_model_tree_5, features_test_1, target_test_1)


# In[117]:


best_result_KNN_5, result_f1_KNN_5, best_model_KNN_5 = KNN(features_train_5, features_val_5, target_train_5, target_val_5)


# In[118]:


acc_KNN_test_5, f1_KNN_test_5 = test_model(best_model_KNN_5, features_test_1, target_test_1)


# In[119]:


#g = sns.heatmap(sqrt_df, annot=True, linewidths=.3, ax=ax, cmap='RdPu');


# In[120]:


table_3 = []
table_4 = []


# In[121]:


append_results_3('DecisionTreeClassifier', acc_tree_test_2, acc_tree_test_4, acc_tree_test_5)
append_results_3('RandomForestClassifier', acc_forest_test_2, acc_forest_test_4, acc_forest_test_5)
append_results_3('9-Nearest Neighbors', acc_KNN_test_2, acc_KNN_test_4, acc_KNN_test_5)
append_results_3('DummyClassifier (most_frequent)', acc_mf_test_1, acc_mf_test_1, acc_mf_test_1)
append_results_3('DummyClassifier (stratified)', acc_s_test_1, acc_s_test_1, acc_s_test_1)


# In[122]:


append_results_4('DecisionTreeClassifier', f1_tree_test_2, f1_tree_test_4, f1_tree_test_5)
append_results_4('RandomForestClassifier', f1_forest_test_2, f1_forest_test_4, f1_forest_test_5)
append_results_4('9-Nearest Neighbors', f1_KNN_test_1, f1_KNN_test_4, f1_KNN_test_5)
append_results_4('DummyClassifier (most_frequent)', f1_mf_test_1, f1_mf_test_1, f1_mf_test_1)
append_results_4('DummyClassifier (stratified)', f1_s_test_1, f1_s_test_1, f1_s_test_1)


# In[123]:


third = pd.DataFrame(table_3)
fourth = pd.DataFrame(table_4)


# In[124]:


third


# In[125]:


third.loc[:,'Do nothing':] = (third.loc[:,'Do nothing':] * 100).round(2) 

#dfi.export(third,"third.png")


# In[126]:


fourth


# In[127]:


fourth.loc[:,'Do nothing':] = (fourth.loc[:,'Do nothing':] * 100).round(2) 

#dfi.export(fourth,"fourth.png")


# In[128]:


# def outlier_plot(data, outlier_method_name, x_var, y_var, 
#                  xaxis_limits=[0,1], yaxis_limits=[0,1]):
    
#     print(f'Outlier Method: {outlier_method_name}')
    
#     # Create a dynamic title based on the method
#     method = f'{outlier_method_name}_anomaly'
    
#     # Print out key statistics
#     print(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
#     print(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
#     print(f'Total Number of Values: {len(data)}')
    
#     # Create the chart using seaborn
#     g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
#     g.map(sns.scatterplot, x_var, y_var)
#     g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
#     g.set(xlim=xaxis_limits, ylim=yaxis_limits)
#     axes = g.axes.flatten()
#     axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
#     axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
#     return g


# In[129]:


# palette = ['#ff7f0e', '#1f77b4']
# sns.pairplot(if_df, vars=anomaly_inputs, hue='anomaly', palette=palette)


# In[130]:


# exp_df = if_df.copy()


# In[131]:


# exp_df = exp_df.loc[exp_df["anomaly"] != -1 ]


# In[132]:


# exp_df


# In[133]:


# exp_df = exp_df.drop(['anomaly_scores', 'anomaly'], axis = 1)


# In[134]:


# import matplotlib.pyplot as plt
# import numpy as np

# # data from https://allisonhorst.github.io/palmerpenguins/

# species = (
#     "1",
#     "2",
#     "3",
# )
# weight_counts = {
#     "Below": np.array([70, best_result_KNN_exp, 58]),
#     "Above": np.array([82, 37, 66]),
#     "Upon":np.array([best_result_KNN_exp,23,45])
# }
# width = 0.5

# fig, ax = plt.subplots()
# bottom = np.zeros(3)

# for boolean, weight_count in weight_counts.items():
#     p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Number of penguins with above average body mass")
# ax.legend(loc="upper right")

# plt.show()


# In[135]:


# import matplotlib.pyplot as plt
# import numpy as np

# # data from https://allisonhorst.github.io/palmerpenguins/

# species = (
#     "Do nothing",
#     "IQR",
#     "SRT",
#     "IF"
# )
# weight_counts = {
#     "DTC": np.array([best_result_tree_2, best_result_tree_4, best_result_tree_5, best_result_tree_exp]),
#     "RFC": np.array([best_result_forest_2, best_result_forest_4, best_result_forest_5, best_result_forest_exp]),
#     "9KNN":np.array([best_result_KNN_2, best_result_KNN_4, best_result_KNN_5, best_result_KNN_exp])
# }
# width = 0.5

# fig, ax = plt.subplots()
# bottom = np.zeros(4)

# for boolean, weight_count in weight_counts.items():
#     p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
#     bottom += weight_count

# ax.set_title("Number of penguins with above average body mass")
# ax.legend(loc="upper right")

# plt.show()


# In[136]:


# X = ['Class 1','Class 2','Class 3', 'Class 4']
# ind = np.arange(len(X)) 
# width = 0.2
# X_axis = np.arange(len(X))

# bar1 = plt.bar(ind, df_target_val.groupby("class_label")["class_label"].count() / len(target_val), width)
# bar2 = plt.bar(ind + width, df_target_test1.groupby("class_label")["class_label"].count() / len(target_test1), width)
# bar3 = plt.bar(ind + width*2, df_target_test2.groupby("class_label")["class_label"].count() / len(target_test2), width)
# bar4 = plt.bar(ind + width*3, df_target_test3.groupby("class_label")["class_label"].count() / len(target_test3), width)

  
# plt.xticks(ind+width,X)
# c = plt.legend( (bar1, bar2, bar3, bar4), ('Val-TX', 'Test1-TX', 'Test2-FL', 'Test3-FL') )
# plt.title('Class label dictribution')
# plt.ylabel('Normalized data set’s counts')
# #fig2 = c.get_figure()


# # Disbalance

# ## Balance = 'weighted'

# In[137]:


#setting the class_weight = balanced, and train the model for the third strategy
best_result_forest_6, result_f1_6, best_model_6 = RFC(features_train_2, features_val_2, target_train_2, target_val_2, 'balanced')


# In[138]:


acc_forest_test_6, f1_forest_test_6 = test_model(best_model_6, features_test_1, target_test_1)


# In[139]:


best_result_tree_6, result_f1_tree_6, best_model_tree_6 = DTC(features_train_2, features_val_2, target_train_2, target_val_2, 'balanced')


# In[140]:


acc_tree_test_6, f1_tree_test_6 = test_model(best_model_tree_6, features_test_1, target_test_1)


# ## Upsampling

# In[141]:


# summarize distribution
counter = Counter(target_train_2)
for k,v in counter.items():
    per = v / len(target_train_2) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# In[142]:


#apply SMOTE to training
oversample = SMOTE(k_neighbors=2)
features_train_2, target_train_2 = oversample.fit_resample(features_train_2, target_train_2)
# summarize distribution
counter = Counter(target_train_2)
for k,v in counter.items():
 per = v / len(target_train_2) * 100
 print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# In[143]:


best_result_forest_7, result_f1_7, best_model_7 = RFC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[144]:


acc_forest_test_7, f1_forest_test_7 = test_model(best_model_7, features_test_1, target_test_1)


# In[145]:


best_result_tree_7, result_f1_tree_7, best_model_tree_7 = DTC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[146]:


acc_tree_test_7, f1_tree_test_7 = test_model(best_model_tree_7, features_test_1, target_test_1)


# In[147]:


best_result_KNN_7, result_f1_KNN_7, best_model_KNN_7 = KNN(features_train_2, features_val_2, target_train_2, target_val_2)


# In[148]:


acc_KNN_test_7, f1_KNN_test_7 = test_model(best_model_KNN_7, features_test_1, target_test_1)


# ## Downsampling

# In[149]:


features_train_2, features_val_2, target_train_2, target_val_2 = preprocess_train(impute_df)


# In[150]:


#let's apply downsampling 
nr = NearMiss()
features_train_2, target_train_2 = nr.fit_resample(features_train_2, target_train_2)
counter = Counter(target_train_2)
for k,v in counter.items():
    per = v / len(target_train_2) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# In[151]:


# train random forest
best_result_forest_8, result_f1_8, best_model_8 = RFC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[152]:


# test random forest
acc_forest_test_8, f1_forest_test_8 = test_model(best_model_8, features_test_1, target_test_1)


# In[153]:


# train decision tree
best_result_tree_8, result_f1_tree_8, best_model_tree_8 = DTC(features_train_2, features_val_2, target_train_2, target_val_2)


# In[154]:


# test decision tree
acc_tree_test_8, f1_tree_test_8 = test_model(best_model_tree_8, features_test_1, target_test_1)


# In[155]:


# train KNN
best_result_KNN_8, result_f1_KNN_8, best_model_KNN_8 = KNN(features_train_2, features_val_2, target_train_2, target_val_2)


# In[156]:


# test KNN
acc_KNN_test_8, f1_KNN_test_8 = test_model(best_model_KNN_8, features_test_1, target_test_1)


# In[157]:


append_results_5('DecisionTreeClassifier', acc_tree_test_6, acc_tree_test_7, acc_tree_test_8)
append_results_5('RandomForestClassifier', acc_forest_test_6, acc_forest_test_7, acc_forest_test_8)
append_results_5('9-Nearest Neighbors', acc_KNN_test_2, acc_KNN_test_7, acc_KNN_test_8)
append_results_5('DummyClassifier (most_frequent)', acc_mf_test_1, acc_mf_test_1, acc_mf_test_1)
append_results_5('DummyClassifier (stratified)', acc_s_test_1, acc_s_test_1, acc_s_test_1)


# In[158]:


append_results_6('DecisionTreeClassifier', f1_tree_test_6, f1_tree_test_7, f1_tree_test_8)
append_results_6('RandomForestClassifier', f1_forest_test_6, f1_forest_test_7, f1_forest_test_8)
append_results_6('9-Nearest Neighbors', f1_KNN_test_2, f1_KNN_test_7, f1_KNN_test_8)
append_results_6('DummyClassifier (most_frequent)', f1_mf_test_1, f1_mf_test_1, f1_mf_test_1)
append_results_6('DummyClassifier (stratified)', f1_s_test_1, f1_s_test_1, f1_s_test_1)


# In[159]:


fifth = pd.DataFrame(table_5)
six = pd.DataFrame(table_6)


# In[160]:


fifth


# In[161]:


six


# In[162]:


fifth.loc[:,'Balance weight':] = (fifth.loc[:,'Balance weight':] * 100).round(2) 

#dfi.export(fifth,"fifth.png")


# In[163]:


six.loc[:,'Balance weight':] = (six.loc[:,'Balance weight':] * 100).round(2) 

#dfi.export(six,"six.png")


# In[ ]:




