import pandas as pd
import numpy as np
import seaborn as sns
import re

import prepare
import ml_categorical
import ml_regression

# nltk is the Natural Language Toolkit. It is a set of libaries to aid with natural language processing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, opinion_lexicon
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk

prepare.prepare()
ml_categorical.rfc()
ml_regression.linear_regression()
ml_regression.pca100()
ml_regression.pca_best()