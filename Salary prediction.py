#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:19:35 2023

@author: noaengel
"""

##Business understanding:
#The project I have chosen is a machine learning salary prediction model. 
#Based on the historical salary data of an imaginary company called DataSolutions 
#I am going to train a model that can predict the future salary of the company’s employees. 
#I am going to use a linear regression model as I have learned it from the modeling classes to try and create a working model. 
#The company would benefit from this as it can save time calculating the salary on their own and makes it fair for employees
#to receive a salary based on their years of experience and their job title/function. 
#I have found the modeling classes and the data mining classes to be very interesting and therefore I want to pursue my skills in these areas. 
#I am certain that there might be some factors of statistics in this as well so it all comes together in my opinion. 

#Data preparation linear regression
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import statsmodels.formula.api as sm
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import statistics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error

#Data preparation
df= pd.read_csv('SalaryData.csv')

print(df.isna().sum())
df.info()
df.describe()
#was not really necessary in this case, as it is a simple dataset

#Preparing the data for training and testing
x = df.drop('Salary',axis=1)
y = df['Salary']

x.shape , y.shape

x_train , x_test , y_train , y_test = train_test_split(x,y,random_state=101,test_size=0.2)
x_train.shape , x_test.shape , y_train.shape , y_test.shape

#utilize the linear regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

#testing the model
prediction= lr.predict(x_test)
prediction
y_test

#making an overview of the actual data, the prediction and the difference between those two
difference= (y_test - prediction)

#Make this look good in a dataframe:
model_outcome= pd.DataFrame(np.c_[y_test , prediction , difference] , columns=['Actual','Predicted','Difference'])
print(model_outcome)

#making a scatterplot of the training data
sns.regplot(x= x_train, y= y_train,lowess=False, color= 'green')

#versus a scatterplot of the testing data
sns.regplot(x= x_test, y= y_test,lowess=False, color= 'red')
#testing data seems to be very linear

#estimating the accuracy of the model
score = lr.score(x_test , y_test)
print (score)
#the score is 0.993422386435995, which is very close to 1.0, 
#this is very good because it means that the model is confident that 
#it is accurate to the mean of the data, if we try this with the r2 score we should get the same number
r2 = r2_score(y_test, prediction)
print(r2)
# and this is correct

#testing the model:
experience = 4 #years
lr.predict([[experience]])[0]
print(f"Salary of {experience} year experience employee = {int(lr.predict([[experience]])[0])} dollar")

#multiple regression model
#data import
salarydf= pd.read_csv('Salary_Data_MP.csv')
#data understanding
salarydf.head()

#data cleaning
print(salarydf.isna().sum())
salarydf.drop(['Race', 'Education', 'gender','otherdetails', 'tag', 'location'],
  axis='columns', inplace=True)

df = salarydf
print(df.isna().sum())
df = df.dropna()
print(df.isna().sum())

dummies= df['title'].astype(str).str.get_dummies()
df = pd.concat([df.drop(columns='title'), dummies], axis=1)

model1= sm.ols('totalyearlycompensation~yearsofexperience+Masters_Degree+Bachelors_Degree+Doctorate_Degree+\
               Race_Asian+Race_White+Race_Two_Or_More+Race_Black+Race_Hispanic+basesalary+stockgrantvalue+bonus', data=df).fit()
print(model1.summary())               

#checking for correlation:
X=df
corr_matrix = X.corr()
print(corr_matrix)
#highest correlation is 0.78, so we check for collinearity
sns.regplot(x='stockgrantvalue', y='totalyearlycompensation', data= df,lowess=True, scatter_kws={"color": "blue"}, line_kws={"color": "green"})
#there seems to be collinearity as more than 90% of the dots are close to the line, that is why I drop stockgrandvalue in the model
model2= sm.ols('totalyearlycompensation~yearsofexperience+Masters_Degree+Bachelors_Degree+Doctorate_Degree+\
                   Race_Asian+Race_White+Race_Two_Or_More+Race_Black+Race_Hispanic+basesalary+bonus', data=df).fit() 
print(model2.summary())
# all of the other values are now significant, but the R-squared value dropped by 0.2
# I want to check if all values are linear, so it might be interesting to standardize them
sns.regplot(x='yearsofexperience', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Masters_Degree', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Bachelors_Degree', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Race_Asian', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Race_White', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Race_Two_Or_More', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Race_Black', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='Race_Hispanic', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='basesalary', y='totalyearlycompensation', data= df,lowess=True)
sns.regplot(x='bonus', y='totalyearlycompensation', data= df,lowess=True)
#There appears to be no non-linear transformation needed
#so, lets standardize continuous variables
scaler = StandardScaler()

scaler.fit(df[['yearsofexperience']])
scaler.fit(df[['basesalary']])
scaler.fit(df[['bonus']])

df['yearsofexperience_scaled'] = scaler.transform(df[['yearsofexperience']])
df['basesalary_scaled'] = scaler.transform(df[['basesalary']])
df['bonus_scaled'] = scaler.transform(df[['bonus']])

model3= sm.ols('totalyearlycompensation~yearsofexperience_scaled+Masters_Degree+Bachelors_Degree+Doctorate_Degree+\
                Race_Asian+Race_White+Race_Two_Or_More+Race_Black+Race_Hispanic+basesalary_scaled+bonus_scaled', data=df).fit() 
print(model3.summary())

#I try to get rid of Outliers by using CooksD, I then mask those outliers from being used in the model
cooksD=model3.get_influence().cooks_distance
n=len(df)
df['Outlier']= cooksD[0] > 4/n

mask=df['Outlier'] == True
df=df.loc[~mask]

#This is the version of the model i will be showing in the report
model4= sm.ols('totalyearlycompensation~yearsofexperience_scaled+Masters_Degree+Bachelors_Degree+Doctorate_Degree+\
                Race_Asian+Race_White+Race_Two_Or_More+Race_Black+Race_Hispanic+basesalary_scaled+bonus_scaled', data=df).fit() 
print(model4.summary())

#Including both models on a html table (in jupyter)
from stargazer.stargazer import Stargazer
from IPython.core.display import HTML
Table = Stargazer([model4])
Table.title('Model Predicting Total Compensation based on multiple factors') #Give it a title   
Table.significant_digits(2)

HTML(Table.render_html())

#Let's try and make a prediction, even though I a not very confident about the outcome of the model
# create a new data frame with the given values
prediction = pd.DataFrame({
    'Years_of_Experience': [12],
    'Masters_Degree': [1],
    'Doctorate_Degree ': [1],
    'Race_Asian  ': [0],
    'Race_White ': [0],
    'Race_Two_Or_More ': [0],
    'Race_Black ': [1],  
    'Race_Hispanic ': [0],
    'Base_Salary': [340000],
    'Bonus': [10000],
    'Bachelors_Degree': [1]
})

# calculating the mean and standard deviation of the original dataframe
mean_experience= statistics.mean(df['totalyearlycompensation'])
std_experience= statistics.stdev(df['totalyearlycompensation'])
mean_basesalary= statistics.mean(df['basesalary'])
std_basesalary= statistics.stdev(df['basesalary'])
mean_bonus= statistics.mean(df['bonus'])
std_bonus= statistics.stdev(df['bonus'])

#standardizing the same values as before
prediction['Base_Salary'] = (prediction['Base_Salary'] - mean_basesalary) / std_basesalary
prediction['Bonus'] = (prediction['Bonus'] - mean_bonus) / std_bonus

#The coefficients from the model:
intercept = 3.89e+07
yearsofexperience_scaled = 5.354e+07
Masters_Degree = -1.145e+04
Bachelors_Degree = -1.266e+04
Doctorate_Degree = 5961.4835
Race_Asian = -5128.7202
Race_White = -1.169e+04
Race_Two_Or_More = -1.333e+04
Race_Black = -1.901e+04
Race_Hispanic = -1.599e+04
basesalary_scaled = 3.353e+04
bonus_scaled = 2.926e+04


#Calculate the predicted value
predicted_value = (
    3.89e+07 +
    (5.354e+07 * prediction['Years_of_Experience']) +
    (-1.145e+04 * prediction['Masters_Degree']) +
    (-1.266e+04 * prediction['Bachelors_Degree']) +
    (5961.4835 * prediction['Doctorate_Degree']) +
    (-5128.7202 * prediction['Race_Asian']) +
    (-1.169e+04 * prediction['Race_White']) +
    (-1.333e+04 * prediction['Race_Two_Or_More']) +
    (-1.901e+04 * prediction['Race_Black']) +
    (-1.599e+04 * prediction['Race_Hispanic']) +
    (3.353e+04 * prediction['Base_Salary']) +
    (2.926e+04 * prediction['Bonus'])
)
print(predicted_value)

#I get an error here that shouldn't be an error^








