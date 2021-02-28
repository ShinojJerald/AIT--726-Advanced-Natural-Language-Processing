# -*- coding: utf-8 -*-

import os
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import re
import numpy as np


def normalize_case(s):    
    '''
    Paramaeter: Word to be normalized
    Converts words with capitalized first letters in to lower case.
    '''
    if(not s.isupper()):
        return s.lower()
    else:
        return s
    
def remove_tags(s):
    '''
    Paramaeter: Word to be normalized
    Removes HTML tags
    '''
    s = re.sub(r'[^\w\s]', '', s)
    return s


def count_words(rootdir):
    '''
    Parameter: root directory
    
    The funtion collects the training files. Tokenizes text into words. Creates stemmed vocabulary and 
    Counts the the occurance of each word in each class(positve and negative).
    '''    
    #Port Stemmer for stemming vocabulary
    ps = PorterStemmer()
    vocab=[]
    total_words=0
    stemmed_vocab=[]
    prior=0
    binary_count={}
    stemmed_binary_count={}
    #For each directory in the path
    for subdir, dirs, files in os.walk(rootdir):
        #For each file in the directory
        for file in files:
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            S = set()
            #For each line in the file
            for line in lines:
                #Tokenize the words using word tokenize of nltk
                document=word_tokenize(line)
                #For each word in the document 
                for i in range(0,len(document)):
                    #Normalize case for the word
                    document[i]=normalize_case(document[i])
                    #Remove HTML tags
                    document[i]=remove_tags(document[i])
                    if(document[i]!=''):
                        total_words+=1
                        #Stem the words and append to stemmed list
                        stemmed_vocab.append(ps.stem(document[i]))
                        if(not document[i] in S):
                            #Calculate stemmed binary count
                            stemmed_binary_count[ps.stem(document[i])]=stemmed_binary_count.get(ps.stem(document[i]),0)+1
                            S.add(ps.stem(document[i]))
                        vocab.append(document[i])
                        if(not document[i] in S):
                            #Calculate  binary count
                            binary_count[document[i]]=binary_count.get(document[i],0)+1
                            S.add(document[i])
            f.close()
            prior+=1
    #Count frequency of words from respective vocabs
    
    count=dict(collections.Counter(vocab))
    stemmed_count=dict(collections.Counter(stemmed_vocab))
    return [count,stemmed_count,prior,binary_count,stemmed_binary_count,total_words]


    

def nb():
    '''
    The funtion collects the count of non stemmed and stemmed vocabulary and assigns it to global variables.
    '''    
    global pos_count
    global neg_count
    global unique_neg_words
    global unique_pos_words
    global stemmed_pos_count
    global stemmed_neg_count
    global positive_prior
    global negative_prior
    global pos_stemmed_binary_count
    global neg_stemmed_binary_count
    
    global pos_binary_count
    global neg_binary_count
    
    global total_pos_words
    global total_neg_words
    #Calculate parameters for positve documents
    rootdir= "train/pos/" 
    count=count_words(rootdir)
    pos_count=count[0]
    stemmed_pos_count=count[1]
    total_pos_words=len(pos_count)
    positive_prior=count[2]
    pos_binary_count=count[3]
    pos_stemmed_binary_count=count[4]
    total_pos_words=count[5]
    
    #Calculate parameters for negative documents
    rootdir= "train/neg/"
    count=count_words(rootdir)
    neg_count=count[0]
    stemmed_neg_count=count[1]
    total_neg_words=len(neg_count)
    negative_prior=count[2]
    neg_binary_count=count[3]
    neg_stemmed_binary_count=count[4]
    total_neg_words=count[5]
    




def get_test():
    '''
    The funtion collects the test data. Creates Bag of words and
    stemmed vocabulary.
    '''    
    #Clean the positive files and create vocab
    ps = PorterStemmer()
    rootdir= "test/pos/"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            document=[]
            tokenized_document=[]
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            for line in lines:
                tokenized=word_tokenize(line)
                final=[]
                tokenized_final=[]
                for token in tokenized:
                    token=normalize_case(token)
                    token=remove_tags(token)
                    final.append(token)
                    tokenized_final.append(ps.stem(token))
                document.append(final)
                tokenized_document.append(tokenized_final)
            test_files.extend(document)
            stemmed_test_files.extend(tokenized_document)
            f.close()
            
    #Clean the negative files and create vocab
    rootdir= "test/neg/"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            document=[]
            tokenized_document=[]
            f=open(rootdir+file,'r',encoding='utf-8') #use the absolute URL of the file
            lines = f.readlines()
            for line in lines:
                tokenized=word_tokenize(line)
                final=[]
                tokenized_final=[]
                for token in tokenized:
                    token=normalize_case(token)
                    token=remove_tags(token)
                    final.append(token)
                    tokenized_final.append(ps.stem(token))
                document.append(final)
                tokenized_document.append(tokenized_final)
            test_files.extend(document)
            stemmed_test_files.extend(tokenized_document)
            f.close()

def classify():
    '''
    The function uses the counts computed from previous step to calculate likelyhood and prior.
    Then the test document is classified based the result from above step.
    '''
    i=1
    #Classify test documents using non-stemmed vocabulary
    for doc in test_files:
        pos_word_likelyhood=1
        neg_word_likelyhood=1
        pos_binary_likelyhood=1
        neg_binary_likelyhood=1
        pos=0
        neg=0
        pos_tfidf=0
        neg_tfidf=0
        for word in doc:
            #Calculate Tf-Idf for the given word
            docFreq=0
            if word in pos_count:
                docFreq+=1
            if word in neg_count:
                docFreq+=1
            if(docFreq>0):
                pos_tfidf=(pos_count.get(word,1)/len(pos_count))/(2/docFreq)
                neg_tfidf=(neg_count.get(word,1)/len(pos_count))/(2/docFreq)
                
            #Calculate likeleyhood for positive and degative documents
            pos_word_likelyhood=pos_word_likelyhood+np.log((pos_count.get(word,1))/(total_pos_words+unique_pos_words+unique_neg_words))
            
            neg_word_likelyhood=neg_word_likelyhood+np.log((neg_count.get(word,1))/(total_pos_words+unique_pos_words+unique_neg_words))
            
            if(word in pos_binary_count):
                pos_binary_likelyhood=pos_binary_likelyhood*(((pos_binary_count.get(word,0))/(2*len(pos_binary_count)+len(neg_binary_count)))+1)
            if(word in neg_binary_count):
                neg_binary_likelyhood=neg_binary_likelyhood*(((neg_binary_count.get(word,0))/(2*len(neg_binary_count)+len(pos_binary_count)))+1)
        #Calculate prior
        pos_class=(pos_word_likelyhood)+np.log(positive_prior/positive_prior+negative_prior)
        neg_class=(neg_word_likelyhood)+np.log(negative_prior/positive_prior+negative_prior)
        #print(pos_class)
        
        #Classify documents based on calculated values
        if(pos_class > neg_class):
            doc_classification[i]='pos'
        elif(pos_class < neg_class):
            doc_classification[i]='neg'
        if(pos_binary_likelyhood>neg_binary_likelyhood):
            binary_doc_classification[i]='pos'
        else:
            binary_doc_classification[i]='neg'
        if(pos_tfidf>neg_tfidf):
            tf_idf_classification[i]='pos'
        else:
            tf_idf_classification[i]='neg'
        i=i+1
    
    #Classify test documents using stemmed vocabulary
    i=1
    for doc in stemmed_test_files:
        pos_word_likelyhood=1
        neg_word_likelyhood=1
        pos_stemmed_binary_likelyhood=1
        neg_stemmed_binary_likelyhood=1
        pos=0
        neg=0
        for word in doc:
            docFreq=0
            if word in pos_count:
                docFreq+=1
            if word in neg_count:
                docFreq+=1
            if(docFreq>0):
                pos_tfidf=(pos_count.get(word,1)/len(pos_count))/(2/docFreq)
                neg_tfidf=(neg_count.get(word,1)/len(pos_count))/(2/docFreq)
                
            pos_word_likelyhood=pos_word_likelyhood+np.log((stemmed_pos_count.get(word,1))/(total_pos_words+unique_pos_words+unique_neg_words))
            neg_word_likelyhood=neg_word_likelyhood+np.log((stemmed_neg_count.get(word,1))/(total_neg_words+unique_pos_words+unique_neg_words))
            if(word in pos_stemmed_binary_count):
                pos_stemmed_binary_likelyhood=pos_stemmed_binary_likelyhood*(((pos_stemmed_binary_count.get(word,0))/(2*len(pos_stemmed_binary_count)+len(pos_stemmed_binary_count))))
            if(word in neg_stemmed_binary_count):
                neg_stemmed_binary_likelyhood=neg_stemmed_binary_likelyhood*(((neg_stemmed_binary_count.get(word,0))/(2*len(neg_stemmed_binary_count)+len(pos_stemmed_binary_count))))
        pos_class=(pos_word_likelyhood)+np.log(positive_prior/positive_prior+negative_prior)
        neg_class=(neg_word_likelyhood)+np.log(negative_prior/positive_prior+negative_prior)
        if(pos_class > neg_class):
            stemmed_doc_classification[i]='pos'
        elif(pos_class <= neg_class):
            stemmed_doc_classification[i]='neg'
        if(pos_stemmed_binary_likelyhood>neg_stemmed_binary_likelyhood):
            binary_doc_classification_stemmed[i]='pos'
        else:
            binary_doc_classification_stemmed[i]='neg'
        if(pos_tfidf>neg_tfidf):
            tf_idf_classification_stemmed[i]='pos'
        else:
            tf_idf_classification_stemmed[i]='neg'
        i=i+1
        
        



def metrics(class_dict):
    '''
    The function calculates accuracy and confusion matrix.
    '''
    arr=np.ndarray(shape=(2,2), dtype=float, order='F')
    arr.fill(0)
    for key in class_dict:
        if(key<=12499 and class_dict[key]=='pos'):
            arr[0][0]=arr[0][0]+1
        if(key<=12499 and class_dict[key]=='neg'):
            arr[0][1]=arr[0][1]+1
        if(key>12499 and class_dict[key]=='pos'):
            arr[1][0]=arr[1][0]+1
        if(key>12499 and class_dict[key]=='neg'):
            arr[1][1]=arr[1][1]+1
            
    class_dict_accuracy=(arr[0][0]+arr[1][1])/len(class_dict)
    print("Accuracy ",class_dict_accuracy)
    print("Confusion Matrix:- ")
    print(arr)
    
def main():
    nb()
    get_test()
    classify()
    print("Non-stemmed Word Count")
    metrics(doc_classification)
    print()
    
    print("Stemmed Word Count")
    metrics(stemmed_doc_classification)
    print()
    
    print("Non-stemmed Binary")
    metrics(binary_doc_classification)
    print()
    
    print("Stemmed Binary")
    metrics(binary_doc_classification_stemmed)
    print()
    
    print("Non-Stemmed TF-IDF")
    metrics(tf_idf_classification)
    print()
    
    print("Stemmed TF-IDF")
    metrics(tf_idf_classification_stemmed)
    
    
if __name__ == "__main__" :
    pos_count={}
    neg_count={}
    
    unique_pos_words=0
    unique_neg_words=0
    
    stemmed_pos_count={}
    stemmed_neg_count={}
    
    positive_prior=0
    negative_prior=0
    
    test_files=[]
    stemmed_test_files=[]
    
    total_neg_words=0
    total_pos_words=0
    
    doc_classification={}
    stemmed_doc_classification={}
    binary_doc_classification={}
    binary_doc_classification_stemmed={}
    tf_idf_classification={}
    tf_idf_classification_stemmed={}
    
    main()

        

