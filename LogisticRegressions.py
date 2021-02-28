# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:32:20 2020

@author: shinoj
"""

import time
start_time = time.time()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import xml
import os
import re
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import warnings
from sklearn import preprocessing
warnings.filterwarnings("ignore")


#reading the text data
pos_train = os.listdir("train/positive/")
neg_train = os.listdir("train/negative/")
pos_test = os.listdir("test/positive/")
neg_test = os.listdir("test/negative/")

# Removing Emoticons
def remove_emoji(string):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251" 
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030""]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string) # no emoji
#Removing HTML tags
def remove_tags(text):
    s = re.sub(r'<[^>]+>', '', text)
    return s

# Cleaning and adding the data
def create_list(dir,type,type1):
    return_list=[]
    for i in range(0, 500):
        file1 = open(type+"/"+type1+"/" + dir[i])
        try:
            text = file1.read()
            text = remove_tags(text)
            text = remove_emoji(text)
            return_list.append(text)
        except UnicodeDecodeError:
            k=0
        file1.close()
    return return_list

#Creating dataframe with positive and negative tweets
def createDataFrame(pos_train_list,neg_train_list):
    df1 = pd.DataFrame(neg_train_list)
    target2 = [0] * len(neg_train_list)
    df1["target"] = target2
    df1 = df1.rename(columns={0: "text"})
#Giving positive tweets as 1 and negative tweets as 0
    df = pd.DataFrame(pos_train_list)
    target1 = [1] * len(pos_train_list)
    df["target"] = target1
    df = df.rename(columns={0: "text"})
#Data is getting shuffled here
    data = pd.concat([df, df1])
    data = shuffle(data,random_state=9)
    x = list(data["text"])
    y=np.array(data["target"])

    return x,y;

# Tokenizing the text
def clean(text):

    vocab=[]
    for j in word_tokenize(text):
        if (j != ''):
            if not j.islower() and not j.isupper():
                j = j.lower()
            vocab.append(j)

    return vocab

# Tokenizing the text and stemming it
def cleaningStemmed(text):

    ps = PorterStemmer()
    vocabulary_stemmed=[]
    for j in word_tokenize(text):
        if (j != ''):
            if not j.islower() and not j.isupper():
                j = j.lower()
            vocabulary_stemmed.append(ps.stem(j))

    return vocabulary_stemmed

#Defining sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# Defining the gradient function for 500 iterations 
def gradient_descent(X, y, params, learning_rate, iterations):

    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost(X, y, params)

    return (cost_history, params)


# The objective cost function
def compute_cost(X, y, theta):

    m = len(y)
    h = sigmoid(X @ theta)
    cost = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))

    return cost

#Defining the regularised gradient function for 500 iterations
def gradient_descent_reg(X, y, params, learning_rate, iterations, lmbda):

    m = len(y)
    cost_history = np.zeros((iterations,1))

    for i in range(iterations):
        params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
        cost_history[i] = compute_cost_reg(X, y, params, lmbda)

    return (cost_history, params)

# The objective regularised cost function
def compute_cost_reg(X, y, theta, lmbda):

    m = len(y)
    h = sigmoid(X @ theta)
    temp = theta
    cost = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lmbda / (2 * m)) * np.sum(np.square(temp))

    return cost


# Final prediction function
def predict(X, params):
    return np.round(sigmoid(X @ params))

# Vectorizing based user input
def vectorizer(X,vectorArr,dict_vocab,vectorType,row,col):

    if (vectorType==1):
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i, dict_vocab[j]] += 1


        idf= np.zeros((row, col), dtype=np.int64)
        for i in range(0,len(vectorArr)):
            for j in range(0,col):
                if vectorArr[i][j] > 0:
                    idf[i][j]= math.log10(row / float(vectorArr[i][j]))

                else:
                    idf[i][j]=0
        vectorArr=np.multiply(vectorArr, idf)

    elif (vectorType==2):
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i, dict_vocab[j]] += 1

    else:
        for i in range(0, len(X)):
            for j in X[i]:
                if j in dict_vocab:
                    vectorArr[i,dict_vocab[j]]=1

    return vectorArr


def main(stemmed,vectorType,regularized):
  # loading the train set
    pos_train_list = create_list(pos_train,"train", "positive")
    neg_train_list = create_list(neg_train,"train", "negative")
    X,y=createDataFrame(pos_train_list,neg_train_list)

    #Checking Stemming/ Not Stemming data
    if stemmed == 1:
        for i in range(0, len(X)):
            X[i] = cleanStemmed(X[i])

    else:
        for i in range(0, len(X)):
            X[i] = clean(X[i])

    # Creating vocabulary list
    vocab = X[0]
    for i in range(1, len(X)):
        vocab.extend(X[i])
    vocab = sorted(set(vocab))

    row = len(X)
    col = len(vocab)

    dict_vocab = {}
    for i, j in enumerate(vocab):
        dict_vocab[j] = i
    trainVector = np.zeros((row, col), dtype=np.int64)

    # Vectorizing the trainset
    trainVector=vectorizer(X,trainVector,dict_vocab,vectorType,row,col)
    m, n = trainVector.shape
    trainVector = np.concatenate([np.ones((m, 1)), trainVector], axis=1)
    trainVector = preprocessing.scale(trainVector)

    initial_theta = np.zeros(n + 1)
    iterations = 1000
    learning_rate = 0.01

    # Logistic function
    if regularized==1:
        lmbda = 0.1
        (cost_history, params_optimal) = gradient_descent_reg(trainVector, y, initial_theta, learning_rate, iterations,lmbda)
    else :
        (cost_history, params_optimal) = gradient_descent(trainVector, y, initial_theta, learning_rate, iterations)

 #Loading the test set
    pos_test_list = create_list(pos_test, "test", "positive")
    neg_test_list = create_list(neg_test, "test", "negative")
    X_test, y_test = createDataFrame(pos_test_list, neg_test_list)

    # Stemming data
    if (stemmed == 1):
        for i in range(0, len(X_test)):
            X_test[i] = cleanStemmed(X_test[i])

    else:
        for i in range(0, len(X_test)):
            X_test[i] = clean(X_test[i])

    row = len(X_test)
    col = len(vocab)

    testVector = np.zeros((row, col), dtype=np.int64)
    # Vectorizing the test data
    testVector = vectorizer(X_test,testVector,dict_vocab,vectorType,row,col)
    m, n = testVector.shape

    testVector=np.concatenate([np.ones((m, 1)), testVector], axis=1)
    testVector = preprocessing.scale(testVector)

    # Final Prediction
    preds = predict(testVector , params_optimal)

    # Final values based on threshold value 0.5
    for i in range(0, len(preds)):
        if (preds[i] <= 0.5):
            preds[i] = 0
        else:
            preds[i] = 1

    if(stemmed==1):
        dataClean="Stemmed"
    else:
        dataClean = "Not Stemmed"

    if (vectorType == 1):
        type = " TF-IDF vectorizer"
    elif (vectorType==2):
        type = " Count vectorizer"
    else:
        type="Binary vectorizer"

    if (regularized==1):
        reg="Regularized"
    else:
        reg="Not regularized"

    # Output
    print("Data Clean: ",dataClean)
    print("Vectorization: ",type)
    print("LinReg costfunction: ",reg)
    print("F1 SCore: ",f1_score(y_test,preds, average='macro'))
    print("Accuracy: ",accuracy_score(y_test,preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print(" ")
if __name__ == "__main__" :
   
    for i in range(1, 3):
        for j in range(1,4):
            for k in range(1,3):
                main(i,j,k)
  #Parameters
        #First Parameter
            #stemmed:1 
            # Not Stemmed: any other number
        #Second Parameter
            #TF-IDF:1
            #Count: 2
            #Binary : any other number
        #Third Parameter
            #regularized=1
            #not regularized=any other number
    #'''

    
    #'''
    #main(0,2,1)
    #main(0, 2, 0)

    
    print("--- %s seconds ---" % (time.time() - start_time))


