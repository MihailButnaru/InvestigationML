import os
import random
import sys
from stop_words import get_stop_words
from sklearn.metrics import accuracy_score, precision_score,f1_score

# Created and Design by MIHAIL BUTNARU
# Naive Bayes Classifier
# It is used to predict the best accuracy of the three classes of food [Indian, Chinese and Turkish]
# It is working bayed on the Bayesiam Theorem P (A | B) = P (B | A) * P (A) / P (B)


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

# Probability of the events P(A | B)
def probabilityEvents(events):
    probabilites = dict()
    total_events = len(events)
    for i in range(0,len(events)):
        if probabilites.get(events[i]):
            probabilites[events[i]] += 1
        else:
            probabilites[events[i]] = 1
    for key in probabilites.keys():
        probabilites[key] /= total_events
    return probabilites

# Get's the LikeliHood for calculating the Bayesian Probability [BAYESIAN THEORY]
def getLikelyhoodTable(data_instances,class_labels,prior):
    cond_prob = dict()
    total_data_instances = len(data_instances)
    for i in range(0, len(data_instances)):
        attributes = getKeyWords(data_instances[i])
        for j in range(0, len(attributes)):
            if cond_prob.get(attributes[j]):
                if cond_prob[attributes[j]].get(class_labels[i]):
                    cond_prob[attributes[j]][class_labels[i]] += 1
                else:
                    cond_prob[attributes[j]][class_labels[i]] = 1
            else:
                cond_prob[attributes[j]] = dict([(class_labels[i], 1)])
    for attr_key in cond_prob.keys():
        for attr_label_key in cond_prob[attr_key].keys() :
            cond_prob[attr_key][attr_label_key] /= (prior[attr_label_key]*total_data_instances)
    return cond_prob

# Prediction fuction that it gets the Data instances, LikeliHood the Prior and total instances of the Training Set

def predict(data_instance,likelihood,prior,total_instances):
    attributes = getKeyWords(data_instance)
    max_posterior = -1
    max_posterior_label = ''
    for label in prior.keys() :
        posterior = prior[label]
        for i in range(0,len(attributes)):
            if likelihood.get(attributes[i]):
                if likelihood[attributes[i]].get(label):
                    posterior *= likelihood[attributes[i]][label]
                    print(posterior)
                else:
                    posterior *= (1/((prior[label]*total_instances)+1))
            else:
                posterior *= (1 / ((prior[label] * total_instances) + 1))
        if posterior > max_posterior :
            max_posterior = posterior
            max_posterior_label = label
    return max_posterior_label

# Gets the Accuracy of the training dataset combined with test dataset
def getAccuracy(data_instances,class_labels,likelihood,prior):
    positive = 0
    total_instances = len(data_instances)
    for i in range (0,len(data_instances)):
        predicted_class = predict(data_instances[i], likelihood, prior,total_instances)
        if class_labels[i] == predicted_class:
            positive +=1
    return (positive/total_instances)*100

# Training Path Link
train_path = sys.argv[1]  
 
# Testing Path Link
test_path = sys.argv[2]    
dir_list = chooseRandomFoodClass(train_path)
print('Randomely Chosen Food classes: ',dir_list)
print('Parsing Train data...')
train_instances, train_class_labels = getLocationClasses(train_path,dir_list)
print(train_instances)
print('Finding prior and likelihood...')
prior = probabilityEvents(train_class_labels)
print(prior)
likelihood = getLikelyhoodTable(train_instances,train_class_labels,prior)
print('Parsing Test data...')
test_instances, test_class_labels = getLocationClasses(test_path,dir_list)
print(test_instances)
accuracy = getAccuracy(train_instances,train_class_labels,likelihood,prior)
print('Accuracy =',accuracy,'%')

