import numpy as nm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Preprocessing
original_mail_data = pd.read_csv('mail_data.csv')

#print(original_mail_data.head())
#print(original_mail_data.isnull().sum())--> No nulls
#print(original_mail_data.shape)

#Label Encoding the spam/ham mail spam is 0 and ham is 1

original_mail_data.loc[ original_mail_data['Category']=='spam', 'Category',]=0
original_mail_data.loc[ original_mail_data['Category']=='ham', 'Category',]=1

#Seperating texts and labels
X = original_mail_data['Message']
Y = original_mail_data['Category']

# print(X)
# print(Y)

#Splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 2)

print(X_train.shape)
print(X_test.shape)
