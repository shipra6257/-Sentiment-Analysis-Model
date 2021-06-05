#!/usr/bin/env python
# coding: utf-8

# # Text Sentiments Model
# 

# ### Author Shipra

# In[55]:


import random


# In[56]:


class Sentiment:
    NEGATIVE="NEGATIVE"
    NEUTRAL="NEUTRAL"
    POSITIVE="POSITIVE"


# In[57]:


class Review:
    def __init__(self,text,score):
        self.text=text
        self.score=score
        self.sentiment=self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <=2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else :
            return Sentiment.POSITIVE
        
        
        
class ReviewContainer:
    def __init__ (self,reviews):
        self.reviews=reviews
        
    def get_text(self):
        return [x.text  for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    def evenly_distribute(self):
        negative=list(filter(lambda x : x.sentiment==Sentiment.NEGATIVE,self.reviews))
        positive=list(filter(lambda x : x.sentiment==Sentiment.POSITIVE,self.reviews))
        positive_shrunk=positive[:len(negative)]
        self.reviews=negative+positive_shrunk
        random.shuffle(self.reviews)
    
                
        


# In[58]:


import json


# In[59]:


file_name= 'C:/Users/asus/Desktop/web development/Books_small_10000b.json'


# In[60]:



        
reviews=[]
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))
        
        
reviews[3].text

    
        


# ## Prep Data

# In[61]:


len(reviews)


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


training,test=train_test_split(reviews,test_size=0.33,random_state=42)

train_container=ReviewContainer(training)
test_container=ReviewContainer(test)
train_container.evenly_distribute()
train_x=train_container.get_text()
train_y=train_container.get_sentiment()
test_container.evenly_distribute()
test_x=test_container.get_text()
test_y=test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))
print(test_y.count(Sentiment.POSITIVE))
print(test_y.count(Sentiment.NEGATIVE))


# In[64]:


print(training[0].sentiment)


# In[65]:


##train_x=[x.text for x in training]
##train_y=[x.sentiment for x in training]

##test_x=[x.text for x in test]
##test_y=[x.sentiment for x in test]


# In[66]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# In[67]:


vectorizer=TfidfVectorizer()

train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors=vectorizer.transform(test_x)


# In[68]:


print(train_x_vectors[0].toarray())


# ## Classification

# In[69]:


## linear svm
from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors,train_y)
clf_svm.predict(test_x_vectors[0])


# In[70]:


## Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf_dec=DecisionTreeClassifier()
clf_dec.fit(train_x_vectors,train_y)
clf_dec.predict(test_x_vectors[0])


# In[71]:


## Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf_gnb=DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors,train_y)
clf_gnb.predict(test_x_vectors[0])


# In[72]:



## logistic Regression
from sklearn.linear_model import LogisticRegression
clf_log=LogisticRegression()
clf_log.fit(train_x_vectors,train_y)
clf_log.predict(test_x_vectors[0])


# ## Evaluation

# In[73]:


## Mean Accuracy
print(clf_svm.score(test_x_vectors,test_y))
print(clf_dec.score(test_x_vectors,test_y))
print(clf_gnb.score(test_x_vectors,test_y))
print(clf_log.score(test_x_vectors,test_y))


# In[74]:


## F1 Score
from sklearn.metrics import f1_score


# In[75]:


f1_score(test_y,clf_svm.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])


# In[76]:


f1_score(test_y,clf_dec.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])


# In[77]:


f1_score(test_y,clf_gnb.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])


# In[78]:


f1_score(test_y,clf_log.predict(test_x_vectors),average=None,labels=[Sentiment.POSITIVE,Sentiment.NEUTRAL,Sentiment.NEGATIVE])


# In[79]:


train_y.count(Sentiment.NEGATIVE)


# In[80]:


train_y.count(Sentiment.POSITIVE)


# In[81]:


test_y.count(Sentiment.POSITIVE)


# In[82]:


test_y.count(Sentiment.NEGATIVE)


# In[95]:


test_set=['this restraunt is not reviewed place']
new_test=vectorizer.transform(test_set)
clf_svm.predict(new_test)


# In[ ]:





# # Tunning our model with grid search

# In[84]:


from sklearn.model_selection import GridSearchCV
parameters={'kernel':('linear','rbf'),'C':(1,4,8,16,32)}
svc=svm.SVC()
clf=GridSearchCV(svc,parameters,cv=5)
clf.fit(train_x_vectors,train_y)


# In[85]:


print(clf.score(test_x_vectors,test_y))


# # Saving Model

# In[86]:


pip install pickle-mixin


# In[87]:


import pickle


# In[88]:


with open('C:/Users/asus/Desktop/web development/model_sentiment.pkl','wb') as f:
    pickle.dump(clf,f)


# # Load model

# In[89]:


with open('C:/Users/asus/Desktop/web development/model_sentiment.pkl','rb') as f:
    loaded_clf=pickle.load(f)


# In[90]:


print(test_x[0])


# In[91]:


loaded_clf.predict(test_x_vectors[0])


# In[100]:


test_set=['Very unstable service, on both phone page and browser']
new_test=vectorizer.transform(test_set)
clf_log.predict(new_test)


# In[ ]:




