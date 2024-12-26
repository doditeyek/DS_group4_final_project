#!/usr/bin/env python
# coding: utf-8

# # Project Title: Used Phone Price Prediction

# # Project Description
# Context
# 
# Buying and selling used phones and tablets used to be something that happened on a handful of online marketplace sites. But the used and refurbished device market has grown considerably over the past decade, and a new IDC (International Data Corporation) forecast predicts that the used phone market would be worth $52.7bn by 2023 with a compound annual growth rate (CAGR) of 13.6% from 2018 to 2023. This growth can be attributed to an uptick in demand for used phones and tablets that offer considerable savings compared with new models.
# 
# Refurbished and used devices continue to provide cost-effective alternatives to both consumers and businesses that are looking to save money when purchasing one. There are plenty of other benefits associated with the used device market. Used and refurbished devices can be sold with warranties and can also be insured with proof of purchase. Third-party vendors/platforms, such as Verizon, Amazon, etc., provide attractive offers to customers for refurbished devices. Maximizing the longevity of devices through second-hand trade also reduces their environmental impact and helps in recycling and reducing waste. The impact of the COVID-19 outbreak may further boost this segment as consumers cut back on discretionary spending and buy phones and tablets only for immediate needs.
# Objective
# 
# The rising potential of this comparatively under-the-radar market fuels the need for an ML-based solution to develop a dynamic pricing strategy for used and refurbished devices. ReCell, a startup aiming to tap the potential in this market, has hired you as a data scientist. They want you to analyze the data provided and build a linear regression model to predict the price of a used phone/tablet and identify factors that significantly influence it.
# Data Description
# 
# The data contains the different attributes of used/refurbished phones and tablets. The detailed data dictionary is given below.

#  # Data Dictionary
# 
# - brand_name: Name of manufacturing brand
# - os: OS on which the device runs
# - screen_size: Size of the screen in cm
# - 4g: Whether 4G is available or not
# - 5g: Whether 5G is available or not
# - main_camera_mp: Resolution of the rear camera in megapixels
# - selfie_camera_mp: Resolution of the front camera in megapixels
# - int_memory: Amount of internal memory (ROM) in GB
# - ram: Amount of RAM in GB
# - battery: Energy capacity of the device battery in mAh
# - weight: Weight of the device in grams
# - release_year: Year when the device model was released
# - days_used: Number of days the used/refurbished device has been used
# - new_price: Price of a new device of the same model in euros
# - used_price: Price of the used/refurbished device in euros

# # Importing Libraries

# In[1]:


## loading and preprocessing data
import pandas as pd 
import numpy as np 

## visualization of data
import matplotlib.pyplot as plt 
import seaborn as sns 

## building validation framework 
from sklearn.model_selection import train_test_split 

# categorical encoding 
from sklearn.feature_extraction import DictVectorizer

## regression model 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

from xgboost import XGBRegressor


## metrics # to compute the accuracy of the model developed
from sklearn.metrics import root_mean_squared_error


# # Loading Dataset

# In[2]:


## loading dataset
data = pd.read_csv('dataset/used_device_data.csv')

## create a copy 
df = data.copy()


# # Data Preview And Understanding

# In[3]:


# view the first rows 
df.head()


# In[4]:


# view the last five rows 
df.tail()


# In[5]:


## check the number of rows and columns 
print(f'Total number of rows: {df.shape[0]}; Total number of columns: {df.shape[1]}')


# In[6]:


# A summary description of the dataset
df.info()


# In[7]:


## checking for missing values
df.isnull().sum()


# In[8]:


## checking for duplicated values 
df.duplicated().sum()


# In[9]:


## checking type on columns
df.dtypes


# In[10]:


## lets return the total counts of unique values in each column 
df.nunique()


# In[11]:


# Checking unique values in each column
for each_name in df.columns: 
    print(each_name)
    print(df[each_name].unique())
    


# # Data preprocessing 
# - normalizing column types
#     - brand_name, os, 4g, and 5g
# - replacing unsual characters with NaN values
# - Filling of missing 
# 

# In[12]:


# changing object type columns to category 
df['brand_name'] = df['brand_name'].astype('category')

df['os'] = df['os'].astype('category') 

df['4g'] = df['4g'].astype('category')

df['5g'] = df['5g'].astype('category')


# In[13]:


# rechecking column types 
df.dtypes 


# In[14]:


# replace empty cells in main_camera_mp column 
df['main_camera_mp'] = df['main_camera_mp'].fillna(0)

# replace empty cells in ram column 
df['ram'] = df['ram'].fillna(0)

# replace empty cells in int_memory column 
df['int_memory'] = df['int_memory'].fillna(0)

# replace empty cells in battery  column 
df['battery'] = df['battery'].fillna(0)

# replace empty cells in weight  column 
df['weight'] = df['weight'].fillna(0)


# In[15]:


# fill in the missing values in main_camera_mp column with mean

df['main_camera_mp'] = pd.to_numeric(df['main_camera_mp'], errors='coerce')
mean_value = df['main_camera_mp'].mean()  
df['main_camera_mp'] = df['main_camera_mp'].fillna(mean_value)

# fill in the missing values in ram column with mean
df['ram'] = pd.to_numeric(df['ram'], errors='coerce')
mean_value = df['ram'].mean()  
df['ram'] = df['ram'].fillna(mean_value)

# fill in the missing values in selfie_camera_mp column with mean
df['selfie_camera_mp'] = df['selfie_camera_mp'].fillna(df['ram'].mean())

# fill in the missing values in int_memory column with mean
df['int_memory'] = df['int_memory'].fillna(df['int_memory'].mean())

# fill in the missing values in battery column with mean
df['battery'] = df['battery'].fillna(df['battery'].mean())

# fill in the missing values in weight column with mean
df['weight'] = df['weight'].fillna(df['weight'].mean())


# In[16]:


# rechecking for missing values in the columns 
df.isnull().sum()


# # Descriptive Analysis

# In[17]:


# Generating statistical summary of the numberic columns 
df.describe().round()


# # Correletation analysis 

# In[18]:


# Correlation results of the target column against other numeric columns
numerical_cols = df.select_dtypes(include=['int', 'float'])

corr_matrix = numerical_cols.corr()

corr_matrix['used_price'].sort_values(ascending=False)


# # Exploratory Data Analysis
# - Target variable analysis
# - Plot a correlation againts the target variable
# - Outlier analysis
# 

# In[19]:


# Plotting a frequency distribution of the target variable 
plt.figure(figsize=(12, 6))

plt.title('Frequency Distribution of Used Phone Price')
plt.xlabel('Price')
plt.ylabel('Count') 

sns.histplot(df['used_price'][df['used_price'] < 100000], color='gold') 

plt.show()


# In[20]:


# log transformation 
# Normalising the skewed data
log_price = np.log1p(df['used_price'])


# In[21]:


# Plotting a frequency distribution of the target variable 
plt.figure(figsize=(12, 7))

plt.title('Frequency Distribution of Used Phone Price')
plt.xlabel('Log Price')
plt.ylabel('Count') 

sns.histplot(log_price, color='magenta') 

plt.show()


# # Scatterplot
# - Correlation between new_price and used_price 

# In[59]:


# plotting new_price and used_price
plt.figure(figsize=(11, 5))

sns.scatterplot(data= df, x='new_price', y='used_price', color='indigo')

plt.title('Correlation Between New Price and Used Phone Price')
plt.xlabel('Price of a New Nevice of the Same Model(€)')
plt.ylabel('Used Price(€)')

plt.show()


# In[53]:


df.head()


# # Box Plot
# - Plotting distribution of used price based on OS

# In[23]:


# plotting distribution of used price based on OS
plt.figure(figsize=(12, 4))

sns.boxplot(x= "os", y="used_price", data=df)

plt.title('Distribution of Used Phone Prices Based on OS on which the Device Runs')
plt.xlabel('Operating System')
plt.ylabel('Used Price(€)')

plt.show()


# # Building a validation framework¶
# - Training dataset 60%
# - Validation dataset 20%
# - Testing dataset 20%

# In[24]:


df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=10)
df_train, df_valid = train_test_split(df_train_full, test_size=0.25, random_state=10)

print(f'Size of Training Dataset: {len(df_train)}')
print(f'Size of Validation Dataset: {len(df_valid)}')
print(f'Size of Testing Dataset: {len(df_test)}')


# In[25]:


## selecting target variable from the dataframe 
y_train = df_train['used_price']
y_valid = df_valid['used_price']


# In[26]:


## deleting the target variable from the dataframe
del df_train['used_price']
del df_valid['used_price']


# In[27]:


# checking if the target variable is deleted from the dataset 
df_train.head()


# # Data Preprocessing 2

# In[28]:


# Converting dataframe to a vector 
# And normalising the target variable 

y_train = np.log1p(y_train).values 
y_valid = np.log1p(y_valid).values 


# In[29]:


#Select columns with int, float and categorial data 
df_train_bl = df_train.select_dtypes(include=['int64', 'float64', 'category'])
df_valid_bl = df_valid.select_dtypes(include=['int64', 'float64', 'category'])


# In[30]:


# convert the dataframe to a dictionary format
dict_train_bl = df_train_bl.to_dict(orient='records')
dict_valid_bl = df_valid_bl.to_dict(orient='records') 

# create an instance of dv
dv = DictVectorizer(sparse=False)

dv.fit(dict_train_bl)


# In[31]:


# transformation 
X_train_bl = dv.transform(dict_train_bl)
X_valid_bl = dv.transform(dict_valid_bl)


# # Training a Baseline Algorithms
# - Linear Regression
# - Decision Tree Regressor
# - Random Forest Regressor
# - Adaboost Regression
# - GradientBoosting
# - Xgboost
# - Bagging Regressor

# # Linear Regression Model

# In[32]:


## creating an instance of a regression 
lr_model = LinearRegression() 

## fit the data to model 
lr_model.fit(X_train_bl, y_train)


# # Model Evaluation

# In[33]:


## generate validation predictions on the validation set  
y_valid_pred = lr_model.predict(X_valid_bl)


# In[34]:


## measure the accuracy 
lr_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 

print(f'Baseline Validation Metric: {lr_rmse_bl}')  

print(f'Baseline Validation Metric: {round(lr_rmse_bl, 2) * 100} %')    # Specify to round it to two decimal places 


# # Decision Tree Regressor

# In[35]:


# create an instance 
dt_model_bl = DecisionTreeRegressor(random_state=11)

dt_model_bl.fit(X_train_bl, y_train)


# In[36]:


# generate validation prediction 
y_valid_pred = dt_model_bl.predict(X_valid_bl)

# Chekcing for accuracy 
dt_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'Decision Tress Baseline Validation Metric: {round(dt_rmse_bl, 2) * 100} %')


# # Random Forest Regressor

# In[37]:


# create an instance 
rf_model_bl = RandomForestRegressor (random_state=11)

rf_model_bl.fit(X_train_bl, y_train)

# generate validation prediction 
y_valid_pred = rf_model_bl.predict(X_valid_bl)

# 
rf_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'Random Forest Baseline Validation Metric: {round(rf_rmse_bl, 2) * 100} %')


# # Adaboost Regressor

# In[38]:


# create an instance 
ar_model_bl = AdaBoostRegressor(random_state=11)

ar_model_bl.fit(X_train_bl, y_train)

# generate validation prediction 
y_valid_pred = ar_model_bl.predict(X_valid_bl)

# 
ar_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'AdaBoost Baseline Validation Metric: {round(ar_rmse_bl, 2) * 100} %')


# # Bagging Regressor

# In[39]:


# create an instance 
br_model_bl = BaggingRegressor(random_state=11)

br_model_bl.fit(X_train_bl, y_train)

# generate validation prediction 
y_valid_pred = br_model_bl.predict(X_valid_bl)

# 
br_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'Bagging Regressor Baseline Validation Metric: {round(br_rmse_bl, 2) * 100} %')


# # Gradient Boosting Regressor

# In[40]:


# create an instance 
gb_model_bl = GradientBoostingRegressor(random_state=11)

gb_model_bl.fit(X_train_bl, y_train)

# generate validation prediction 
y_valid_pred = gb_model_bl.predict(X_valid_bl)

# 
gb_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'Gradient Boosting Baseline Validation Metric: {round(gb_rmse_bl, 2) * 100} %')


# # XGBoost

# In[41]:


# create an instance 
xgb_model_bl = XGBRegressor(random_state=11)

xgb_model_bl.fit(X_train_bl, y_train)

# generate validation prediction 
y_valid_pred = xgb_model_bl.predict(X_valid_bl)

# 
xgb_rmse_bl = root_mean_squared_error(y_valid, y_valid_pred) 
print(f'XGBoost Regressor Baseline Validation Metric: {round(xgb_rmse_bl, 2) * 100} %')


# # Training and Testing Final Model

# In[44]:


df_train_fm = df_train_full.select_dtypes(include=['int64', 'float64', 'category'])
df_test_fm = df_test.select_dtypes(include=['int64', 'float64', 'category'])

#Normalise and transform to a vector
y_train_fm = np.log1p(df_train_full['used_price']).values 
y_test_fm = np.log1p(df_test['used_price']).values 

# del target column from dataset 
del df_train_fm['used_price']
del df_test_fm['used_price']

# convert the dataframe to a dictionary format
dict_train_fm = df_train_fm.to_dict(orient='records')
dict_test_fm = df_test_fm.to_dict(orient='records') 

# training a dictVector
dv = DictVectorizer(sparse=False)
dv.fit(dict_train_fm)

# transform to matrix format
X_train_fm = dv.transform(dict_train_fm)
X_test_fm = dv.transform(dict_test_fm)

# creating an instance of a regression 
gb_model_fm = GradientBoostingRegressor(random_state=11)

# training the model
gb_model_fm.fit(X_train_fm, y_train_fm)

# generate validation predictions on the test set  
y_test_pred_fm = gb_model_fm.predict(X_test_fm)

## measure the accuracy 
gb_rmse_fm = root_mean_squared_error(y_test_fm, y_test_pred_fm) 

print(f'Gradient Boosting Model Final Test Metric: {round(gb_rmse_fm, 2) * 100} %')    # Specify to round it to two decimal places 


# # Save the Model

# In[46]:


import pickle 
with open('used_phone_Price_Prediction.bin', 'wb') as f_out: 
    pickle.dump((dv, gb_model_fm), f_out) 


# # Loading the mode

# In[47]:


with open('used_phone_Price_Prediction.bin', 'rb') as f_in:
    dv.model = pickle.load(f_in)
    


# # Making Predictions

# In[75]:


device_data = {
 'brand_name': input('Enter name of manufacturing brand (e.g., Samsung, Nokia): '),
 'os': input('Enter operating system on which your device runs (e.g., Android, iOS): '),
 'screen_size': float(input('Enter the size of the screen in cm: ')),
 '4g': input('Does the device support 4G? (yes/no): '),
 '5g': input('Does the device support 5G? (yes/no): '),
 'main_camera_mp': float(input('Enter the resolution of the rear camera in megapixels: ')),
 'selfie_camera_mp': float(input('Enter the resolution of the front camera in megapixels: ')),
 'int_memory': float(input("What is your device's internal memory (in GB)?: ")),
 'ram': float(input('Enter the RAM (in GB): ')),
 'battery': float(input('Enter the energy capacity of the device battery (in mAh): ')),
 'weight': float(input('Enter the weight of the device (in grams): ')),
 'release_year': int(input('In which year was the device model released?: ')),
 'days_used': int(input('How many days have you used the device?: ')),
 'new_price': float(input('Enter the price of a new device of the same model (in Euros): ')),
 }

# # print(f'\n--SPECIFICATIONS--')
# print(f'\n -- DEVICE SPECIFICATIONS-- \n\n {device_data}')

# printing the device data in dictionary format
print('\n --DEVICE SPECIFICATIONS-- \n')
for key, value in device_data.items():
    print(f'{key}: {value}')


# In[76]:


## lets create a function to make a single prediction 
def predict_single(df, dv, gb_model_fm):
    X = dv.transform([device_data])
    y_pred = gb_model_fm.predict(X)[0]
    return y_pred

## lets call the function to make the prediction 
predicted_log_price = predict_single(device_data, dv, gb_model_fm)

## output the value of the prediction 
print(f'The log price of the used device is €{predicted_log_price.round(2)}')


# In[77]:


# reversing log price to the original price 
actual_price = np.expm1(predicted_log_price)
print(f'Your used/refurbished device can be sold for:  €{actual_price.round(2)}')


# In[ ]:




