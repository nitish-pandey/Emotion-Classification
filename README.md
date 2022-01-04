
# Emotion Analysis

>Project status : Active and Incomplete


## Table of content :
* **Project intro**

        1. General info

        2. Methods used

        3. Technologies used

        4. Setup

        5. Dataset



* **Project Description**
* **Codes and technical aspects**
* **Deployment**

# 1. Project introduction

## General info

This project creates the machine learning model that helps us to predict whether the given text message refers to which emotion ( like anger , sadness,love , etc.)

## Methods used

   * Machine Learning

   * Natural language processing

   * Predictive modeling

   * etc.


## Technologies used

This project is created using [**python**](https://www.python.org/) and other libraries like :

* [Pandas](https://pandas.pydata.org/)
* [Numpy](https://numpy.org/)
* [Sklearn/Scikit-Learn](https://scikit-learn.org/stable/#)
* [NLTK](https://www.nltk.org/)
* etc.

![image](https://user-images.githubusercontent.com/85100877/143047907-f5b9f36f-35d9-41b9-8bfe-d9ed224bf642.png)


The other librarires are enlisted in the requirement.txt file.

## Setup

 
To run this project , install [**python**](https://www.python.org/) locally and install the requirements using [pip](https://pypi.org/project/pip/) or [conda](https://docs.conda.io/en/latest/) -

```terminal
	pip install -r requirement.txt
```

else you can use the [google colab notebook](Spam_Classifier.ipynb) for running the code online without installing any libraries , packages and dependencies.

![image](https://user-images.githubusercontent.com/85100877/143045414-3468cf84-395e-4ad5-8f2c-cc7ab878b3f7.png)


## Dataset


The dataset used in this project is available in [Kaggle](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp)

Collection of documents and its emotions, It helps greatly in NLP Classification tasks
List of documents with emotion flag, Dataset is split into train, test & validation for building the machine learning model

Example :-
i feel like I am still looking at a blank canvas blank pieces of paper;sadness



# 2.Project Description



# 3.Codes and Technical aspects

This project is entirely based on the machine learning and Natural language processing.

For that , we will be using **Python** Language as it is suitable and contains various libraries and packages  , and has easy syntax.
Alternatively , **R** language can also be used.


For running the project , python must be installed in your local system.
The required packages and libraries are listed in [requirement.txt](requirement.txt).

You can also use **Google-Colab** . **Colaboratory**, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education.
The colab notebook is also uploaded here.


The code simply contains multiple parts :

First of all ,we have to import all the required libraries like nltk, sklearn , etc.

And then , we have to import our dataset which contains the messages with the label of spam or not spam.

We have our dataset in the **txt** format . The each line in txt file contains one message along with the corresponding emotion separated by " ; " .
We import our data using **pandas** library and save as the dataframe. 


The message we have in our dataset might not be clean . So we have to remove some unwanted stuffs like stopwords (i.e. "just" ,"is","oh","an" ,etc. ) and we also have to reduce the word into the word root form ( i.e. playing,plays,played to play, ).
we can do it so by using **Lamatization process**. The **NLTK** library provides the tool for that.


After this , we have to convert the words into vectors because the machine learning algorithm can't be directly fed with strings or characters . To we have to convert the sentences into vectors (i.e. numeric matrix )
This process is called feature extraction.
There are multiple tools for that like **Countvectorizer ,TF-IDF, word2vec**,etc. 
We are here using **TF-IDF Vectorizer** which is one the widely used one.



In the machine learning part , we are using **Logistic Regression** algorithm to create a model.
This model preforms better in these cases and is also economical (in the case of time , memory and computational cost) than others.

We create an object of Logistic Regression  algorithm and train it using the training dataset( both messages and labels ).


we predict the value for the testing messages (at this time only messages are passed not their labels) and compare with the original value/labels .
By this the proformance of the model is analyzed using various metrices like accuracy , confusion matrix , classification report , etc.


If the performance of the model is good . The model is ready to use and can be saved. 
Else the model needs to be re-trained ( by using another algorithm or by parameter tuning .)


Both model and the vectorizer needs to be saved for the deployment or future use.




# 4.Deployment

Hence , the model is created using machine learning . The model needs to be deployed for its practical use.

We create a Web-App using the [Flask](https://flask.palletsprojects.com/en/2.0.x/). 

![image](https://user-images.githubusercontent.com/85100877/143044908-a797ef8b-cfd6-41fe-a33f-390eb16c9111.png)

Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries.


The sample Deployment web page looks like :: 
