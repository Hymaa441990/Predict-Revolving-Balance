# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:02:56 2020

@author: Hymavathi Samsani
"""

import pickle
import numpy as np
import pandas as pd

# Splitting data into training and testing
from sklearn.model_selection import train_test_split

from sklearn import linear_model

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


#----------Find what the charecter encoding-----
#import chardet

file="Data - Copy.csv"
#with open(file,'rb') as rawdata:
#    result=chardet.detect(rawdata.read(100000))
#result

#Charecter encoding done in ascii

df = pd.read_csv(file,encoding='ISO8859-1',low_memory=False,sep=',')
df.columns=df.columns.str.rstrip() #Remove white spaces in column names
df.columns=df.columns.str.lstrip()
#print(df.head())

#Data Types
#display(df.info())

# Statistics for each column
#display(df.describe())

#Drop unqiue ID features
df=df.drop(columns=['member_id','batch_ID'],axis=1)

# --------------------Rename the reveloving balance ------------------------------------------
df = df.rename(columns = {'total revol_bal': 'total_revol_bal'})

# Statistics for each column
desc_temp=df.describe()
#display(desc_temp)

#Drop below features as 25th to 75% percetile has zero values only.
#i.e., except rest all numbers fall under outliers and there is no use if tyy to apply any outlier handling technique as 75 to 80% data has zero

features_with_Most_zeros=['pub_rec', 'delinq_2yrs','total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_colle_amt']
df=df.drop(columns=features_with_Most_zeros,axis=1)
#display(df.head())
#display(df.shape)

#---------Data Munging--------------------------
# Missing Values
# Rename columns
# Data Transformation
# Coding,en-coding,decoding
# Outlier
# Anamolies

#display(df.isnull().head())

#Missing Values
#display(df.isnull().sum())

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
    
missing_values_table(df)

#---------------- Get the columns with > 50% missing-------------
missing_df = missing_values_table(df);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
#print('We will remove %d columns.' % len(missing_columns))

# ---------Drop the columns----------------
df = df.drop(columns = list(missing_columns))
#display(df.head())
#display(df.shape)

#Dropping 29 records which is having blank data for most of the features
df=df[df['inq_last_6mths'].isnull() == False]
final_missing_columns=missing_values_table(df)
#print(final_missing_columns.index)

#display(missing_values_table(df))

X=df.drop('total_revol_bal',axis=1)
y=df['total_revol_bal'].copy()
#display(df.head())
#display(X.head())

#80% Data Split - train(70%) and Test(30%)
XX_train, XX_test, yy_train, yy_test = train_test_split(X,y,test_size=0.3,random_state=27)
#display(XX_train.shape)
#display(XX_test.shape)
#display(yy_train.shape)
#display(yy_test.shape)

# --------------------Create List for each type ------------------------------------------
def Creat_List_For_Each_Type(df,num_cols):
    int_vars=[]
    categorical_vars=[]
    continuous_vars=[]

    for i in df.columns:
        if num_cols.columns.__contains__(i):
           if num_cols[i]['25%']+num_cols[i]['50%']+num_cols[i]['75%'] == 0 :
               int_vars.append(i)
           else:
               continuous_vars.append(i)
        else:
            categorical_vars.append(i)
    return int_vars,categorical_vars,continuous_vars
      
int_vars,categorical_vars,continuous_vars=Creat_List_For_Each_Type(XX_train,desc_temp)
#print("Categorical Variables: "+','.join(categorical_vars)+"\n\nInteger Variables: "+','.join(int_vars)+"\n\nContinuous Variables: "+','.join(continuous_vars))



#Dropping below features as per the previous analysis

Keep_cols=['tot_curr_bal','annual_inc','debt_income_ratio','numb_credit','loan_amnt','total_credits','home_ownership','Emp_designation','total_rec_int','verification_status','initial_list_status','Rate_of_intrst','Experience','purpose']
#len(Keep_cols)

XX_train=XX_train[Keep_cols]
#len(XX_train.columns)

#display(len(XX_train))
desc_temp=XX_train.describe()
int_vars,categorical_vars,continuous_vars=Creat_List_For_Each_Type(XX_train,desc_temp)
#print("Categorical Variables: "+','.join(categorical_vars)+"\n\nInteger Variables: "+','.join(int_vars)+"\n\nContinuous Variables: "+','.join(continuous_vars))



def RemoveOutliers(df,var_name):
    # Calculate first and third quartile
    first_quartile = df[var_name].describe()['25%']
    third_quartile = df[var_name].describe()['75%']

    # Interquartile range
    iqr = third_quartile - first_quartile
    #print(iqr)
    # Remove outliers
    df=df[(df[var_name] >= (first_quartile - (1.15*iqr))) &
                (df[var_name] <= (third_quartile + (1.15*iqr)))]
    return df
    
def ImputeOutliers(df_t,var_name):
    q1 = df_t[var_name].quantile(0.25) #first quartile value
    q3 = df_t[var_name].quantile(0.75) # third quartile value
    iqr = q3-q1 #Interquartile range
    low  = q1-1.5*iqr #acceptable range
    high = q3+1.5*iqr #acceptable range
    
    #display(var_name)
    df_t[var_name]=df_t.query(str(var_name)+' >= @low')[var_name]
    df_t[var_name].fillna(value=low, inplace=True)
    #display("Low - "+str(low))
    
    df_t[var_name]=df_t.query(str(var_name)+' <= @high')[var_name]
    df_t[var_name].fillna(value=high, inplace=True)
    #display("High - "+str(high))
    
    return df_t

def outlier(df_t,var_name):
    from numpy import percentile
    # seed the random number generator    
    # generate univariate observations
    # calculate interquartile range
    q1 = df_t[var_name].quantile(0.25) #first quartile value
    q3 = df_t[var_name].quantile(0.75) # third quartile value
    iqr = q3-q1 #Interquartile range
    low  = q1-1.5*iqr #acceptable range
    high = q3+1.5*iqr #acceptable range
    #print(i)
    #print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q1, q3, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower, upper = q1 - cut_off, q3 + cut_off
    # identify outliers
    outliers = [x for x in df_t[var_name] if x < low or x > high]
    ##print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in df_t[var_name] if x >= low and x <= high]
    ##print('Non-outlier observations: %d' % len(outliers_removed))
    
    return [len(outliers),len(outliers_removed),high]

final_missing_columns=missing_values_table(XX_train)
#print(final_missing_columns.index)

#display(missing_values_table(XX_train))


#Missing Values Imputation
XX_train['Emp_designation'].fillna(value=XX_train['Emp_designation'].value_counts().index[0], inplace=True)
XX_train['Experience'].fillna(value=XX_train['Experience'].value_counts().index[0], inplace=True)
XX_train['tot_curr_bal'].fillna(value=XX_train['tot_curr_bal'].median(), inplace=True)

final_missing_columns=missing_values_table(XX_train)
#print(final_missing_columns.index)

#display(missing_values_table(XX_train))


for i in continuous_vars:
   XX_train=ImputeOutliers(XX_train,i)

home_ownership=pd.get_dummies(XX_train['home_ownership'],drop_first=False)
#display(home_ownership.head())

verification_status=pd.get_dummies(XX_train['verification_status'],drop_first=False)
#display(verification_status.head())

initial_list_status=pd.get_dummies(XX_train['initial_list_status'],drop_first=False)
#display(initial_list_status.head())

Experience=pd.get_dummies(XX_train['Experience'],drop_first=False)
#display(Experience.head())

purpose=pd.get_dummies(XX_train['purpose'],drop_first=False)
#display(purpose.head())

#Drop Categorical Features
XX_train=XX_train.drop(['home_ownership','verification_status','initial_list_status','Experience','purpose'],axis=1)
XX_train.head()

XX_train=pd.concat([XX_train,home_ownership,verification_status,initial_list_status,Experience,purpose],axis=1)
#display(XX_train.head())

XX_train = XX_train.rename(columns = {'< 1 year': 'Lessthan1Year'})
XX_train = XX_train.rename(columns = {'10+ years': '10PlusYears'})
#display(XX_train.describe())

XX_train=XX_train.drop(columns='Emp_designation',axis=1)
XX_train

xgb_regressor=xgb.XGBRegressor()

regressor = xgb.XGBRegressor(alpha=15, base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.3, gamma=0.3, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.3, max_delta_step=0, max_depth=5,
             min_child_weight=7, missing=np.nan, monotone_constraints='()',
             n_estimators=100, n_jobs=0, num_parallel_tree=1,
             objective='reg:squarederror', random_state=0, reg_alpha=15,
             reg_lambda=10, scale_pos_weight=1, subsample=0.5,
             tree_method='exact', validate_parameters=1, verbosity=None)

regressor.fit(XX_train,yy_train)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
