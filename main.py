import nltk
import pandas as pd
import warnings 
import string
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


nltk.download()
warnings.filterwarnings("ignore")
stopwords = stopwords.words("english")
stemmer = PorterStemmer()


#read file with data
fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")
fake_data['True/Fake'] = 0
true_data['True/Fake'] = 1

#print basic informations about datasets
print("****** Fake data ******")
print(fake_data.head(), "\n")
print("****** True data ******")
print(true_data.head())

#concatenate data 
data = pd.concat([fake_data, true_data])

print("****** Linked data ******")
print(data.shape)
print(data.describe())
print(data.info())

#remove stopwords, stemming
def clean_text(text):
    list_of_words = [stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", text).split() if i not in stopwords]
    return " ".join(list_of_words).lower()


data["text"] = data["text"].apply(lambda text: clean_text(text))

#split data on train and test datasets

target = data["True/Fake"]
X_train, X_test, y_train, y_test = train_test_split(data["text"], target, test_size=0.20, random_state=100)

print("****** Shape of datasets ******")
print(data.shape); print(X_train.shape); print(X_test.shape)

#converting text to word frequency vectors
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
train_tfIdf = vectorizer_tfidf.fit_transform(X_train.values.astype('U'))
test_tfIdf = vectorizer_tfidf.transform(X_test.values.astype('U'))

print("****** Shape of vectors with converted text ******")
print(train_tfIdf.shape); print(test_tfIdf.shape)

#create model

nb_classifier = MultinomialNB()
nb_classifier.fit(train_tfIdf, y_train)

#predicting test values, display some of them and accuracy
pred = nb_classifier.predict(test_tfIdf) 
print(pred[:10])

accuracy_tfidf = metrics.accuracy_score(y_test, pred)
print(accuracy_tfidf)

Conf_metrics_tfidf = metrics.confusion_matrix(y_test, pred, labels=[1, 0])
print(Conf_metrics_tfidf)



