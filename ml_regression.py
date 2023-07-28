# Regression with PCA

# The purpose of this code is to demonstrate linear regression

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def prepare_data():
    # Prepare the data set
    data_new = pd.read_csv("New_7k_clean_data.csv")
    data_new.drop(columns="Comments Text",inplace=True)
    data_new.drop(columns="Target",inplace=True)

    cv = CountVectorizer(lowercase=True)
    data_array = cv.fit_transform(data_new.Comments_clean).toarray()
    x, x_test, y, y_test = train_test_split(data_array, data_new.TargetNum, random_state=42, test_size=0.1)
    return x, x_test, y, y_test

def linear_regression():
    x, x_test, y, y_test = prepare_data()
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    
    y_pred = regr.predict(x)
    print("R^2 on training data: ",r2_score(y,y_pred))
    y_pred_test = regr.predict(x_test)
    print("R^2 on test data: ",r2_score(y_test, y_pred_test))

def pca100():
    x, x_test, y, y_test = prepare_data()
    # PCA with 100 features
    pca = PCA(n_components=100)
    pca.fit(x)
    # The data set needs to be expressed in terms of the principle components
    x_train_transform = pca.transform(x)
    x_test_transform = pca.transform(x_test)

    # Fit a linear model, as above, but this time with the principle components
    regr = linear_model.LinearRegression()
    regr.fit(x_train_transform,y)

    y_pred = regr.predict(x_train_transform)
    print("R^2 for the training data: ", r2_score(y,y_pred))
    y_pred_test = regr.predict(x_test_transform)
    print("R^2 for the test data: ", r2_score(y_test, y_pred_test))

    # Running the above, it can be seen that R^2 on the test set is not great, but at least it is not
    # the trivial value observed from a full linear model without PCA.

def pca_best():
    x, x_test, y, y_test = prepare_data()
    # These lists will contain the R^2 values for various numbers of principle components.
    results_train = []
    results_test = []

    # Up to 1000 components. This may take a while to run.
    pca1000 = PCA(n_components=1000)
    pca1000.fit(x)
    x_train_transform = pca1000.transform(x)
    x_test_transform = pca1000.transform(x_test)

    for i in range(1,1001):
        # Apply a linear model on the first i principle components and store the R^2 values for test and train.
        regr = linear_model.LinearRegression()
        regr.fit(x_train_transform[:,:i],y)
        y_pred = regr.predict(x_train_transform[:,:i])
        results_train.append(r2_score(y,y_pred))
        y_pred_test = regr.predict(x_test_transform[:,:i])
        results_test.append(r2_score(y_test, y_pred_test))

    # Save the image
    plt.plot(results_train)
    plt.plot(results_test)
    plt.xlabel("Number of principle components")
    plt.ylabel("$R^2$")
    plt.annotate(xy=(500,0.62),text="Training",rotation=16)
    plt.annotate(xy=(450,0.46),text="Test",rotation=0)
    plt.savefig('pca.png')
    plt.show()
    plt.close()