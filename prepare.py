# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk is the Natural Language Toolkit. It is a set of libaries to aid with natural language processing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, opinion_lexicon
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk

##############################

# Strip out numbers and punctuation, regularize the spacing.
# The following does not affect capitalization.
def remove(x,stopword,lem):
    x=re.sub("[^A-z" "]+"," ",x)
    s=""
    for i in x.split():
        if i not in stopword:
            i=lem.lemmatize(word=i)
            s=s+" "+i
    return s

# Return the sentiment as one of three values: positive, negative, or neutral.
def polarity(sent, vs):
    d=vs.polarity_scores(sent)
    if d["compound"]>0.05:
        return "positive"
    elif d["compound"]<-0.05:
        return "negative"
    else:
        return "neutral"

# Sentiment, kept as a numerical value between -1 and +1.
def polarity_num(sent, vs):
    return vs.polarity_scores(sent)["compound"]

def prepare():
    # Uncomment these lines if these packages have not been downloaded.
    # They should only need to be run once per python installation.
    #nltk.download('omw-1.4')
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    #nltk.download('opinion_lexicon')

    data=pd.read_excel('Depression & Anxiety Facebook page Comments Text.xlsx')

    stopword = stopwords.words("english")
    lem=WordNetLemmatizer()
    data["Comments_clean"] = data["Comments Text"].apply(remove,stopword=stopword,lem=lem)

    vs = SentimentIntensityAnalyzer()
    data["Target"] = data["Comments_clean"].apply(polarity, vs=vs)
    data["TargetNum"] = data["Comments_clean"].apply(polarity_num, vs=vs)

    data.to_csv("New_7k_clean_data.csv",index=False)