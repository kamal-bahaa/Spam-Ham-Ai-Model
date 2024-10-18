from tkinter import *
import tkinter as tk

# %% [markdown]
# ## Important libraries

# %%
import math
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import nltk 
import pickle 
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer 
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Downloading NLTK data
"""
nltk.download('punkt')       # Downloading tokenizer data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('wordnet')
"""

# %% [markdown]
# ## Read and show the data

# %%
df=pd.read_csv("spam_ham_dataset.csv")
#df.head()


# %% [markdown]
# ## Cleaning the data

# %%
df.drop(["label"],axis=1,inplace=True) #drop unnecessary column
#df.head()

# %%
df.isnull().sum() #no missing values in all rows

df["# sent emails "].isnull().sum() #no null number of emails sent
df["text"].isnull().sum() #no null emails

# %%
df[df["# sent emails "]<0].sum() #no negative number of emails

# %%
df["text"].duplicated().sum() #check duplicate email

# %%
df.drop_duplicates(subset=["text"],inplace=True) #drop duplicated emails

# %%
df['label_num'].unique()

# %% [markdown]
# ## show information about the data

# %%
#df.info()

# %%
#df.describe()

# %% [markdown]
# ## Preprocessing for the data 

# %%
def preprocess(email):
    email=re.sub("^Subject: ","",email) #remove (Subject: ) 
    email=re.sub("[^a-zA-Z]"," ",email) #remove special characters
    #email=re.sub("\s\w\s","",email) # remove s in 's and t in 't (like book's cover or he can't)
    email=re.sub("^\s+","",email) #remove leading space
    email=re.sub("\s+$","",email) #remove trailing space
    email=re.sub("\s+"," ",email) #remove extra spaces between words
    email=email.lower()           #lowercase every word
    
    return email
#test
"""
x=preprocess(df.loc[2,"text"])
x
"""


# %%
def tokenize(email):
     list_of_words=nltk.word_tokenize(email)
     return list_of_words
#test
"""
listt=tokenize("hey   tokeniz .a.tion 7amada     bro !   ?")
listt
"""

# %%
def remove_stopwords(email):
   clean_words=[]
   list_of_words=tokenize(email)

   for word in list_of_words:
      if(word not in stopwords.words('english')):
         clean_words.append(word)
         
   email=' '.join(clean_words) #convert list to string with seperator between every element (' ')

   return email  
  #test
"""
x=preprocess(df.loc[14,"text"])
xnew=remove_stopwords(x)
print(tokenize(xnew))
"""



# %%
def lemmatize_email(email):
    lemmatized_words=[]
    
    lemmatizer=WordNetLemmatizer()
    for word in tokenize(email):
        new_word=lemmatizer.lemmatize(word)
        lemmatized_words.append(new_word)
    email=' '.join(lemmatized_words)
    return email

#test
"""
x=preprocess(df.loc[14,"text"])
xnew=remove_stopwords(x)
xnew2=lemmatize_email(xnew)
xnew2
"""


# %%
def stem_email(email):
    stemmed_words=[]
    
    stemmer=PorterStemmer()
    for word in tokenize(email):
        new_word=stemmer.stem(word)
        stemmed_words.append(new_word)
    email=' '.join(stemmed_words)
    return email


# %%
def from_email_column_to_list_of_emails(df):
    list_emails=[]
    for i in range(len(df)):
        list_emails.append(df["text"].iloc[i])
    return list_emails    



# %%
def from_label_column_to_list_of_labels(df):
    list_labels=[]
    for i in range(len(df)):
        list_labels.append(df["label_num"].iloc[i])
    return list_labels
    

# %%
def clean_text(text):
    text=preprocess(text)
    text=remove_stopwords(text)
    text=lemmatize_email(text)
    return text

# %%
def spam_or_ham(text):
    
    cleaned_text = clean_text(text)
    # Clean the text
    listt=[]
    
    listt.append(cleaned_text)
    
    # Transform the cleaned text using the pre-fitted vectorizer
    x_pred=vectorizer.transform(listt)
    
# Predict whether the text is spam or ham
    flag = lrc.predict(x_pred)
    return flag


# %%
# apply the functions of preprocessing on the text column

df['text']=df['text'].apply(preprocess)
df['text']=df['text'].apply(remove_stopwords)
df['text']=df['text'].apply(lemmatize_email)


# %%
#df["text"].iloc[0]

# %%

vectorizer=TfidfVectorizer()
#vectorizer=CountVectorizer() #Another way of feature extraction

X=vectorizer.fit_transform(df["text"])


Y=df["label_num"]
#X and Y for Logistic Regression
lrc=LogisticRegression()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
lrc.fit(X_train,Y_train)
Y_pred = lrc.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)







# %%
#test spam_or_ham function
"""
for i in range(100):
 flag=spam_or_ham(df["text"].iloc[i])
 print(df["label_num"].iloc[i])
 print(flag)
"""
 

# %% [markdown]
# ## Visualization of data

# %%
sns.barplot(x='label_num',y='# sent emails ',data=df) #number of sent emails by spam/ham emails

# %%
sns.boxplot(x='label_num',y='# sent emails ',data=df)

# %%
sns.displot(data=df,x="# sent emails ") #number of sent emails count histogram

# %%
sns.countplot(data=df,x=df["label_num"]) #count spam/ham

# %%





def read_email():
    text=textbox.get("1.0",tk.END)
    check=spam_or_ham(text)
    if(check==1):
        result_label.config(text="spam")
    else:
        result_label.config(text="ham")


   
root=tk.Tk()

root.geometry("800x800")

root.title("Spam Email Classifier")

label=tk.Label(root,text="Check emails for free !",font=('Arial',18))
label.pack()

textbox=tk.Text(root,font=('Arial',12))
textbox.pack(padx=100,pady=100)

button1=tk.Button(root,text="Check",font=('Arial',18),command=read_email)
button1.pack(padx=10,pady=10)

result_label=tk.Label(root,text="",font=("Arial",18))
result_label.pack()

root.configure(bg="#013E58")


root.mainloop()

