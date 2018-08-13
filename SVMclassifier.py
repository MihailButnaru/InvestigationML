import numpy as np
import pandas as pd
import sys
from sklearn import datasets
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.externals import joblib
from nltk.stem import *
from nltk.stem.porter import *
import os
import random
from stop_words import get_stop_words

# Created and Design by MIHAIL BUTNARU
# SUPPORT VECTOR MACHINES

#Gets the key words and it's splitting every word with a space [FACEBOOK POSTS]
def getKeyWords(data_instance):
    wordList = data_instance.split(" ")
    return list(set(wordList))

# Randomly will choose 3 classes of FOOD from the training set in order to improve the running time of the algorithm.

def chooseRandomFoodClass(path):
    dataList = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    dataList = random.sample(dataList, 3)           # 3 FOOD classes (Indian / Chinese / Turkish)
    return dataList

# Function to remove unwanted symbols / words from the posts
def removeStopWords(text):
    stop_words = get_stop_words('en')
    words = getKeyWords(text)
    resultwords = [word for word in words if word.lower() not in stop_words]
    result = ' '.join(resultwords)
    result = ''.join(e for e in result if e.isalpha() or e == ' ')
    result = " ".join(result.split())
    return result

# Checks for the Header of each file in order to know from where to start to analyse the posts. [LINES: ]
def startingPoint(lines):
    delimiter = 'Lines:'
    i = 0
    for i in range(0,len(lines)):
        if delimiter in lines[i]:
            break
    return ' '.join(lines[i+1:])

# Function for reading every line from the FILE, the text is in English [LATIN - encoding]
def readLinesFromFile(path):
    with open(path, 'r' , encoding='latin-1') as myfile:
        text = myfile.read()
    lines = text.split('\n')
    return lines
    
# Get's the correct path of the Training and Test set [Must be declared in the when the file is runnning.]
def getLocationClasses(dirpath,dirlist):
    attribut_list = []
    class_list = []
    for dir_name in dirlist:
        path_name = dirpath + dir_name + '/'
        for file in os.listdir(path_name):
            lines = readLinesFromFile(path_name + '/' + file)
            text = startingPoint(lines)
            text = removeStopWords(text)
            attribut_list.append(text)
            class_list.append(dir_name)
    return attribut_list,class_list

if __name__ == '__main__':
  sizes = []
  # The Name of the Training Path
  train_path = '/Users/MichaelButnaru/Desktop/DocumentC/train/'
  # The Name of the Test Path
  test_path = '/Users/MichaelButnaru/Desktop/DocumentC/test/'
  dir_list = chooseRandomFoodClass(train_path)
  train_instances, train_class_labels = getLocationClasses(train_path,dir_list)
  docs_test = train_instances

  # The vectorizer and transformer libraries extractas the features from the trained data.
  counter = CountVectorizer()
  train_data = counter.fit_transform(train_instances)
  tf_transformer = TfidfTransformer(use_idf=False).fit(train_data)
  train_tf = tf_transformer.transform(train_data)
  tfidf_transformer = TfidfTransformer()
  train_tfidf = tfidf_transformer.fit_transform(train_data)

  # Every word from a sentence is getting split it and analysed individually.
  stemmer = PorterStemmer()
  words = []
  st = []
  for i in range(len(train_instances)):
    words = train_instances[i].split(" ")
    singles = [stemmer.stem(word) for word in words]
    st.append(' '.join(singles))
  
  print("SVM")

  # The data is getting trained and analysed, the testing data is compared with the trained data.
  text_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge',penalty='l2'))
                      ])

  text_train = text_svm.fit(st, train_instances)
  predicted = text_train.predict(docs_test)
  dataAccuracy = metrics.accuracy_score(train_instances, predicted)
  precision = metrics.precision_score(train_instances, predicted, average='weighted')
  recall = metrics.recall_score(train_instances, predicted, average = "macro")
  print("Accuracy:", dataAccuracy)
  print("Precision: ", precision)
  print("Recall", recall)



