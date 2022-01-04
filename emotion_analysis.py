# -*- coding: utf-8 -*-
"""Emotion-Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Hg2TKSPhWyjQZJDEyHaQxuRiD-A7_WtQ

#Imports
"""

import pandas as pd
import numpy as np
import os
import random
import re


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


"""# Data / Paragraph"""

train=pd.read_csv(r'C:\Users\20065\OneDrive\Documents/train.txt',sep=';',names=['Sentences','Emotion'])
test=pd.read_csv(r'C:\Users\20065\OneDrive\Documents/test.txt',sep=';',names=['Sentences','Emotion'])
val=pd.read_csv(r'C:\Users\20065\OneDrive\Documents/val.txt',sep=';',names=['Sentences','Emotion'])


train.drop_duplicates(inplace=True)
train.dropna(inplace=True)


"""#Data Cleaning"""

from nltk.corpus import stopwords
stoplist=stopwords.words('english')

from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

def expand(phrase):
    
    phrase = re.sub(r"wont", "will not", phrase)
    phrase = re.sub(r"wouldnt", "would not", phrase)
    phrase = re.sub(r"shouldnt", "should not", phrase)
    phrase = re.sub(r"couldnt", "could not", phrase)
    phrase = re.sub(r"cudnt", "could not", phrase)
    phrase = re.sub(r"cant", "can not", phrase)
    phrase = re.sub(r"dont", "do not", phrase)
    phrase = re.sub(r"doesnt", "does not", phrase)
    phrase = re.sub(r"didnt", "did not", phrase)
    phrase = re.sub(r"wasnt", "was not", phrase)
    phrase = re.sub(r"werent", "were not", phrase)
    phrase = re.sub(r"havent", "have not", phrase)
    phrase = re.sub(r"hadnt", "had not", phrase)

    
    phrase = re.sub(r"n\ t", " not", phrase)
    phrase = re.sub(r"\re", " are", phrase)
    phrase = re.sub(r"\ s ", " is ", phrase) 
    phrase = re.sub(r"\ d ", " would ", phrase)
    phrase = re.sub(r"\ ll ", " will ", phrase)
    phrase = re.sub(r"\dunno", "do not ", phrase)
    phrase = re.sub(r"ive ", "i have ", phrase)
    phrase = re.sub(r"im ", "i am ", phrase)
    phrase = re.sub(r"i m ", "i am ", phrase)
    phrase = re.sub(r" w ", " with ", phrase)

    return phrase

def process(sentences):
    list=[]
    for i in range(len(sentences)):

        # Removing all characters except alphabets
        temp=re.sub('[^a-zA-Z]',' ',sentences[i])
        
        #Expanding the word ( like wont into will not )
        temp=expand(temp)

        #lowering all characters
        temp=temp.lower()
        #splitting the sentences into words
        temp=temp.split()

        # lamaetizing the only words which are not present in stopwords
        temp=[lemmatizer.lemmatize(word) for word in temp if word not in set(stoplist)]
    
        # joining the words into sentences
        temp=' '.join(temp)

        # Appending the new sentences into the new list which will be forward proceeded
        list.append(temp)
    return list

train['Sentences']=process(np.array(train['Sentences']))
test['Sentences']=process(np.array(test['Sentences']))
val['Sentences']=process(np.array(val['Sentences']))

emotion=np.array(train['Emotion'].unique())
dict={}
for i,e in enumerate(emotion):
    dict[e]=i


train['Emotion']=train['Emotion'].replace(dict)
test['Emotion']=test['Emotion'].replace(dict)
val['Emotion']=val['Emotion'].replace(dict)



"""#Word Embedding

##TF-IDF
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_features=8000)
train_data=tfidf.fit_transform(train['Sentences'])
test_data=tfidf.transform(test['Sentences'])
val_data=tfidf.transform(val['Sentences'])

train_label=train.Emotion.values
test_label=test.Emotion.values
val_label=val.Emotion.values

"""# Machine Learning Implementation"""

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(max_iter=100000)

clf.fit(train_data,train_label)

pred=clf.predict(test_data)

from sklearn.metrics import classification_report ,confusion_matrix,accuracy_score
print('Accuracy : ',accuracy_score(test_label,pred))
print('\nConfusion Matrix : \n',confusion_matrix(test_label,pred))
print('\n\nClassification Report : ',classification_report(test_label,pred))

import matplotlib.pyplot as plt
import seaborn as sns
label=['Sadness','Anger','Love','Surprise','Fear','Joy']
matrix=confusion_matrix(test_label,pred)

matrix=pd.DataFrame(matrix,columns=label,index=label)
fig, ax = plt.subplots(figsize=(15,15))
ax.set(title='Confusion Matrix')

sns.heatmap(matrix,cmap='Blues',annot=True,ax=ax)

"""#Saving the model"""

import joblib

joblib.dump(clf,'mymodel.pkl')
joblib.dump(tfidf,'TF-IDF_vectorizer.pkl')