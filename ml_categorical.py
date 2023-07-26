import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def rfc():
    # Load and prepare the data
    data_new = pd.read_csv("New_7k_clean_data.csv")
    data_new.drop(columns="Comments Text",inplace=True)
    cv = CountVectorizer(lowercase=True)
    data_array = cv.fit_transform(data_new.Comments_clean).toarray()

    # Random Forest Classifier
    x, x_test, y, y_test = train_test_split(data_array, data_new.Target, random_state=42, test_size=0.1)
    rf = RandomForestClassifier()
    rf.fit(x,y)
    pred=rf.predict(x_test)
    print("Accuracy: ",accuracy_score(y_test, pred)*100)
    print("")
    print("Classification Report:\n",classification_report(y_test, pred))
    print("")
    print("Confusion Matrix:\n",confusion_matrix(y_test, pred))

    print(data_new.head())