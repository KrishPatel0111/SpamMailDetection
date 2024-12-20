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
Y = original_mail_data['Category'].astype('int')

# print(X)
#print(Y)

#Splitting data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state = 2)

# print(X_train.shape)
# print(X_test.shape)

#transform the text data to feauture vectors
feature_extract  = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_feature = feature_extract.fit_transform(X_train)
X_test_feature = feature_extract.transform(X_test)

#print(X_train_feature)
#training logistic regression model
model = LogisticRegression()
model.fit(X_train_feature, Y_train)

#Evaluation
#Prediction on training data
prediction_on_train = model.predict(X_train_feature)
accuracy_on_train = accuracy_score(Y_train, prediction_on_train)
print('Accuracy on training data : ', accuracy_on_train)

#Prediction on test data
prediction_on_test = model.predict(X_test_feature)
accuracy_on_test = accuracy_score(Y_test, prediction_on_test)
print('Accuracy on test data : ', accuracy_on_test)

#no overfitting
#predictive system
while True:
  input_mail  = [input("Enter the mail: ")]
  input_data_feature = feature_extract.transform(input_mail)
  input_prediction = model.predict(input_data_feature)
  if input_prediction[0]==0:
    print("This is a spam email")
  elif input_prediction[0]==1:
    print("This is a ham email")
  else:
    print("Some Error Occured")
  