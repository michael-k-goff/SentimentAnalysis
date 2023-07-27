# Regression with PCA

# The purpose of this code is to demonstrate linear regression

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.metrics import r2_score

def linear_regression():
    cv = CountVectorizer(lowercase=True)
    data_new = pd.read_csv("New_7k_clean_data.csv")
    data_new.drop(columns="Comments Text",inplace=True)
    data_new.drop(columns="Target",inplace=True)

    data_array = cv.fit_transform(data_new.Comments_clean).toarray()
    x, x_test, y, y_test = train_test_split(data_array, data_new.TargetNum, random_state=42, test_size=0.1)
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    
    y_pred = regr.predict(x)
    print("R^2 on training data: ",r2_score(y,y_pred))
    y_pred_test = regr.predict(x_test)
    print("R^2 on test data: ",r2_score(y_test, y_pred_test))

# To be done: PCA