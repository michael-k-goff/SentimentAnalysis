# Sentiment Analysis

This is a short project to illustrate sentiment analysis. The project demonstrates the following:

- Text Preprocessing with the NLTK (Natural Language Toolkit) library,
- Lemmatizing,
- Sentiment Analysis with [VADER](https://github.com/cjhutto/vaderSentiment) (Valence Aware Dictionary and sEntiment Reasoner),
- Count Vectorization,
- Prediction of VADER's sentiment label with a random forest classifier,
- Prediction of VADER's sentiment score with linear regression,
- Principle Component Analysis,
- Evaluation of the model with precision, recall, and f1-score by label, as well as a confusion matrix.

The Jupyter notebook contains more explanatory comments and more code displaying intermediate output than the python scripts.

The file `pca.png` portrays R^2^ values for principle components analyses up to 1000 components. While the training set shows continually improving R^2^, demonstrating improvement of fit, the test set shows a plateau at around 150 components and an eventual decline, indicating overfitting. Thus the optimum number of components is around 150.