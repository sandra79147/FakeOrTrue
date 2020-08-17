import nltk
import pandas as pd 
import string
import re
import warnings 

warnings.filterwarnings("ignore")

nltk.download()


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

#removing puncatuations and tokenization
def clean_text(text):
    symbols = list(text)
    removed_punctuation = "".join([char for char in symbols if char not in string.punctuation])
    new_text = re.split("\W+", removed_punctuation)
    return new_text

data["text"] = data["text"].apply(lambda text: clean_text(text))












