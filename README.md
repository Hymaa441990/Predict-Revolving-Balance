# Predict-Revolving-Balance

<b>Business Objective:</b><br>
Revolving credit means you're borrowing against a line of credit. Let's say a lender extends a certain amount of credit to you, against which you can borrow repeatedly. The amount of credit you're allowed to use each month is your credit line, or credit limit. You're free to use as much or as little of that credit line as you wish on any purchase you could make with cash. Its just like a credit card and only difference is they have lower interest rate and they are secured by business assets.
At the end of each statement period, you receive a bill for the balance. If you don't pay it off in full, you carry the balance, or revolve it, over to the next month and pay interest on any remaining balance. As you pay down the balance, more of your credit line becomes available and usually its useful for small loans
As a bank or investor who are into this revolving balance here they can charge higher interest rates and convenience fees as there is lot of risk associated in customer paying the amount. Our company wants to predict the revolving balance maintained by the customer so that they can derive marketing strategies individually.

<b>Data Set Details:</b>    
    
This dataset consists of <b>800000</b> observations    
    
<b>member_id</b> unique ID assigned to each member    
<b>loan_amnt</b> loan amount ($) applied by the member    
<b>terms:</b>  term of loan (in months)    
<b>batch_ID</b> batch numbers allotted to members    
<b>Rate_of_intrst</b>:  interest rate (%) on loan    
<b>Grade</b>:grade assigned by the bank    
<b>sub_grade</b>: grade assigned by the bank    
<b>emp_designation</b> job / Employer title of member    
<b>Experience</b>: employment length, where 0 means less than one year and 10 means ten or more years    
<b>home_ownership</b> status of home ownership    
<b>annual_inc</b>: annual income ($) reported by the member    
<b>verification_status</b> status of income verified by the bank    
<b>purpose</b> purpose of loan    
<b>State</b>: living state of member    
<b>debt-to-income ratio</b> : ratio of member's total monthly debt    
<b>Delinquency of past 2 years</b>:  ( failure to pay an outstanding debt by due date)    
<b>inq_6mths</b>: Inquiries made in past 6 months    
<b>total_months_delinq</b> : number of months since last delinq    
<b>Nmbr_months_last_record</b>: number of months since last public record    
<b>Numb_credit_lines</b>:number of open credit line in member's credit line    
<b>pub_rec</b>: number of derogatory public records    
<b>Tota_credit_revolving_balance</b>: total credit revolving balance    
<b>total_credits</b>: total number of credit lines available in members credit line    
<b>list_statu</b>s unique listing status of the loan - W(Waiting),F(Forwarded)    
<b>int_rec</b>: Total interest received till date    
<b>late_fee_rev</b>: Late fee received till date    
<b>recov_chrg</b>: post charge off gross recovery    
<b>collection_recovery_fee</b> post charge off collection fee    
<b>exc_med_colle_12mon</b>: number of collections in last 12 months excluding medical collections    
<b>since_last_major_derog</b>: months since most recent 90 day or worse rating    
<b>application_type</b> indicates when the member is an individual or joint    
<b>verification_status_joint</b> indicates if the joint members income was verified by the bank    
<b>last_pay_week</b>: indicates how long (in weeks) a member has paid EMI after batch enrolled    
<b>nmbr_acc_delinq</b>: number of accounts on which the member isdelinquent    
<b>colle_amt</b>: total collection amount ever owed    
<b>curr_bal</b>: total current balance of all accounts    

import numpy as np    
import pandas as pd    
import seaborn as sns    
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder    
import warnings    
warnings.filterwarnings("ignore")    
    
from sklearn.svm import SVC    
from sklearn.metrics import confusion_matrix    
    
# No warnings about setting value on copy of slice    
pd.options.mode.chained_assignment = None    
    
# Display up to 60 columns of a dataframe    
pd.set_option('display.max_columns', 60)    
    
# Matplotlib visualization    
import matplotlib.pyplot as plt    
#%matplotlib inline    
    
# Set default font size    
plt.rcParams['font.size'] = 24    
    
# Internal ipython tool for setting figure size    
from IPython.core.pylabtools import figsize    
    
# Seaborn for visualization    
import seaborn as sns    
sns.set(font_scale = 2)    
    
# Splitting data into training and testing    
from sklearn.model_selection import train_test_split    
    
#Missing values    
import missingno as mno    
    
from sklearn import linear_model    
