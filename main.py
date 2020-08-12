
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib as plt

fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

#print basic informations about datasets

print("****** Fake data ******")
print(fake_data.head())
print("****** True data ******")
print(true_data.head())

data = pd.concat([fake_data, true_data])


print("****** Linked data ******")
print(data.shape)
print(data.describe())
print(data.info())

#clean data and display as numeric value or vectors 












