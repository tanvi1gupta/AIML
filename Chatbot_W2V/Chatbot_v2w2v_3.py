
# coding: utf-8

# In[ ]:

from textblob import TextBlob
#from attributegetter import *
from generatengrams import ngrammatch
from Contexts import *
import json
from Intents import *
import random
import os
import re
import pandas as pd
import copy


# In[ ]:

import numpy as np
#import tflearn
import numpy as np
import pandas as pd
import nltk 
from nltk import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging

#use the existing google bin to get word2vec representation
#create own word 2 vec
#perform word2vec on these words
num_features = 300    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 6           # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
isUseZeros = True
isUseOnes = False

model = gensim.models.KeyedVectors.load_word2vec_format('../../Datasets/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)


# In[ ]:

#using stemmer to remove and clean sentences for word2vec
#nltk.download('stopwords')
stemmer = SnowballStemmer("english")
#stemmer = LancasterStemmer()

stopwords = set(stopwords.words('english'))
#print(stopwords)
only_alnum = re.compile(r"[^\w]+") ## \w => unicode alphabet

def clean_sentence(sentence):
    sentence = re.sub(only_alnum, " ", sentence).strip()
    words = nltk.word_tokenize(sentence)
    return words


# In[ ]:

#import the intent file
import json
intents = []
files = os.listdir('./intents/')
intents = {}
documents = []

#some utility methods
def convertClassToNumber(intent):
    if intent == 'BookRestaurant':
        return 1
    elif intent == 'movieChoice':
        return 0
    return -1

def convertNumberToClass(intentNumber):
    if intentNumber == 1:
        return 'BookRestaurant'
    elif intentNumber == 0:
        return 'movieChoice'
    return None

for fil in files:
    lines = open('./intents/'+fil, encoding='windows-1252').readlines()
    intent_lines = []
    for i, line in enumerate(lines):
        line = line.lower()
        words = clean_sentence(line)
        intent_lines.append(words)
        documents.append((convertClassToNumber(fil[:-4]), words))
    intents[fil[:-4]] = intent_lines



# In[ ]:

#method to convert all documents into word2vec representation
def convert_sentence2vec(documents, isUseZeros, isUseOnes):
    missing_word_vecs={}
    number_of_docs = len(documents)
    word2vec_rep = np.zeros((number_of_docs, num_features))
    count = 0 
    labels = []
    i=0
    for document in documents:
        tag = document[0]
        doc_words = document[1]
        #print(doc_words)
        for word in doc_words: 
            try:
                word2vec_rep[i]+=model[word]
            except:
                '''The word isn't in our pretrained word-vectors, hence we add a random gaussian noise
                    to account for this. We store the random vector we assigned to the word, and reuse 
                    the same vector during test time to ensure consistency.'''
                if word  not in missing_word_vecs.keys():
                    if isUseZeros:
                        missing_word_vecs[word] = np.zeros((num_features))
                    elif isUseOnes:
                        missing_word_vecs[word] = np.ones((num_features))
                    else:
                        missing_word_vecs[word] = np.random.normal(-0.25, 0.25, num_features)
                word2vec_rep[i]+=missing_word_vecs[word]
                count +=1
        labels.append(tag)
        i+=1
    return word2vec_rep, labels, missing_word_vecs


# In[ ]:
#if a new test sample is received, use this method to compute word2vec represenation
def word2vec_representation(doc_words):
    word2vec_rep = np.zeros((1, num_features))
    for word in doc_words: 
        try:
            word2vec_rep+=model[word]
        except:
            '''The word isn't in our pretrained word-vectors, hence we add a random gaussian noise
                    to account for this. We store the random vector we assigned to the word, and reuse 
                    the same vector during test time to ensure consistency.'''
            if word  not in missing_word_vecs.keys():
                if isUseZeros:
                    missing_word_vecs[word] = np.zeros((num_features))
                elif isUseOnes:
                    missing_word_vecs[word] = np.ones((num_features))
                else:
                    missing_word_vecs[word] = np.random.normal(-0.25, 0.25, num_features)
            word2vec_rep+=missing_word_vecs[word]
    return word2vec_rep
            
    


# In[ ]:
#converting the intents into word2vec
sent2vec_rep, labels, missing_word_vecs = convert_sentence2vec(documents, True, False)


# In[ ]:

#KNN classification problem
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3, weights='uniform')
classifier.fit(sent2vec_rep, labels)


# In[ ]:

def predict_class(sentence):
    w2v1 = word2vec_representation(clean_sentence(sentence))
    return convertNumberToClass(int(classifier.predict(w2v1))),classifier.predict_proba(w2v1)
    


# In[ ]:
#loading the csv files which have the relation between differents entities per intent
moviedataset = pd.read_csv('corpus/moviedata.csv')
moviedataset = moviedataset.apply(lambda x: x.astype(str).str.lower())
moviedataset[['size']] = moviedataset[['size']].apply(pd.to_numeric)


restaurantDataset = pd.read_csv('corpus/restaurantBooking.csv')
restaurantDataset = restaurantDataset.apply(lambda x: x.astype(str).str.lower())
restaurantDataset[['size']] = restaurantDataset[['size']].apply(pd.to_numeric)


# In[ ]:

def check_actions(current_intent, attributes, context):
    '''This function performs the action for the intent
    as mentioned in the intent config file'''
    '''Performs actions pertaining to current intent
    for action in current_intent.actions:
        if action.contexts_satisfied(active_contexts):
            return perform_action()
    '''

    context = IntentComplete()
    return 'action: ' + current_intent.action, context

def check_required_params(current_intent, attributes, context, inferred_attributes):
    '''Collects attributes pertaining to the current intent'''
    #print(attributes)
    all_attributes = copy.deepcopy(attributes)
    inferred_attributes_copy = copy.deepcopy(inferred_attributes)
    all_attributes.update(inferred_attributes_copy)
    for para in current_intent.params:
        #print(para.name)
        if para.required:
            if para.name not in all_attributes:
                #Example of where the context is born, implemented in Contexts.py
                if para.name=='mname':
                    context = MovieName()
                elif para.name == 'mlocation':
                    context = MovieLocation()
                elif para.name == 'language':
                    context = Language()
                elif para.name == 'genre':
                    context = Genre()
                elif para.name == 'actor':
                    context = Actor()
                elif para.name == 'theatre':
                    context = Theatre()
                elif para.name == 'time':
                    context = Time()
                elif para.name == 'date':
                    context = Date()
                elif para.name == 'numberOfPeople':
                    context = NumberOfPeople()
                elif para.name == 'seattype':
                    context = SeatType()
                
                #returning a random prompt frmo available choices.
                return random.choice(para.prompts), context

    return None, context


def input_processor(user_input, context, attributes, intent):
    '''Spellcheck and entity extraction functions go here'''
    
    #user_input = TextBlob(user_input).correct().string
    
    #update the attributes, abstract over the entities in user input
    attributes, cleaned_input = getattributes(user_input, context, attributes)
    
    return attributes, cleaned_input

def loadIntent(path, intent):
    with open(path) as fil:
        dat = json.load(fil)
        intent = dat[intent]
        #print(intent)
        return Intent(intent['intentname'],intent['Parameters'], intent['actions'])

def intentIdentifier(clean_input, context,current_intent):
    clean_input = clean_input.lower()
    #print(clean_input)
    #Scoring Algorithm, can be changed.
    scores = ngrammatch(clean_input)
    
    #choosing here the intent with the highest score
    scores = sorted_by_second = sorted(scores, key=lambda tup: tup[1])
    
    #using KNN to get the intent class
    intentclass, probabilities = predict_class(clean_input)
    #printing the probabilities of ngram and KNN
    print(scores, intentclass, probabilities)
    if(current_intent==None):
        if(clean_input=="movie"):
            return loadIntent('params/newparams.cfg', 'movieChoice')
        elif(clean_input=="restaurant"):
            return loadIntent('params/newparams.cfg', 'BookRestaurant')
        else:
            return loadIntent('params/newparams.cfg', intentclass)
    else:
        #If current intent is not none, stick with the ongoing intent
        return current_intent

def getattributes(uinput,context,attributes):
    '''This function marks the entities in user input, and updates
    the attributes dictionary'''
    #Can use context to context specific attribute fetching
    if context.name.startswith('IntentComplete'):
        return attributes, uinput
    else:
        #Code can be optimised here, loading the same files each time suboptimal 
        files = os.listdir('./entities/')
        entities = {}
        for fil in files:
            lines = open('./entities/'+fil, encoding='windows-1252').readlines()
            for i, line in enumerate(lines):
                line = line.lower()
                lines[i] = line[:-1]
            entities[fil[:-4]] = '|'.join(lines)

        #Extract entity and update it in attributes dict
        for entity in entities:
            for i in entities[entity].split('|'):
                if i.lower() in uinput.lower():
                    attributes[entity] = i
        for entity in entities:
                uinput = re.sub(entities[entity],r'$'+entity,uinput,flags=re.IGNORECASE)

        #Example of where the context is being used to do conditional branching.
        #if 'mname' in attributes  or (context.name=='MovieBooking_moviename' and context.active):
        #    match = attributes['mname']
        #    context.active = False
        return attributes,uinput


# In[ ]:

class Session:
    def __init__(self, attributes=None, active_contexts=[FirstGreeting(), IntentComplete() ]):
        
        '''Initialise a default session'''
        
        #Active contexts not used yet, can use it to have multiple contexts
        self.active_contexts = active_contexts
        
        #Contexts are flags which control dialogue flow, see Contexts.py        
        self.context = FirstGreeting()
        
        #Intent tracks the current state of dialogue
        #self.current_intent = First_Greeting()
        self.current_intent = None
        
        #attributes hold the information collected over the conversation
        self.attributes = {}
        self.inferred_attributes = {}
        
    def update_contexts(self):
        '''Not used yet, but is intended to maintain active contexts'''
        for context in self.active_contexts:
            if context.active:
                context.decrease_lifespan()

    def reply(self, user_input):
        '''Generate response to user input'''
        
        #graceful shutdown
        if 'bye' in user_input.lower() or 'thank' in user_input.lower() or 'quit' in user_input.lower() or 'awesome' in user_input.lower() :
            self.attributes = {}
            self.inferred_attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None
            return 'Thanks for chatting, hope you have a good day'
        
        self.attributes, clean_input = input_processor(user_input, self.context, self.attributes, self.current_intent)
        self.current_intent = intentIdentifier(clean_input, self.context, self.current_intent)
        
        #constructing query to run on the data set
        
        #extracting out implicit entities from the given inputs and querying the corpus to get the results
        isDFEmpty = False
        if self.attributes:
            if self.current_intent.name =='movieChoice':
                isDFEmpty, self.inferred_attributes,qry = add_unique_attributes(moviedataset, self.attributes)
            elif self.current_intent.name =='BookRestaurant':
                isDFEmpty, self.inferred_attributes,qry = add_unique_attributes(restaurantDataset, self.attributes)

		#if the resultant dataframe is empty, then return error and reset
        if isDFEmpty == True and self.current_intent.name =='movieChoice':
            prompt = 'No tickets available for '+ str(self.attributes) +'\n\n Please retry'
            self.attributes = {} 
            self.inferred_attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None
            return prompt
        if isDFEmpty == True and self.current_intent.name =='BookRestaurant':
            prompt = 'No restaurant table available for '+ str(self.attributes) +'\n\n Please retry'
            self.attributes = {}
            self.inferred_attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None
            return prompt
        prompt, self.context = check_required_params(self.current_intent, self.attributes, self.context, self.inferred_attributes)
        #prompt being None means all parameters satisfied, perform the intent action
        
        if prompt is None:
            if self.context.name!='IntentComplete':
                prompt, self.context = check_actions(self.current_intent, self.attributes, self.context)
                prompt = prompt + '\n'+ str(self.attributes) + str(self.inferred_attributes) +'\n\n'+ 'Thank you'
        
        #Resets the state after the Intent is complete
        if self.context.name=='IntentComplete':
            self.attributes = {}
            self.inferred_attributes = {}
            self.context = FirstGreeting()
            self.current_intent = None
        elif self.context.name == 'MovieBooking_moviename':
            prompt = prompt +  '\n'+ str(moviedataset.query(qry)) 
        elif self.context.name == 'RestaurantBooking_restaurantNames':
            prompt = prompt +  '\n'+ str(restaurantDataset.query(qry)) 

            
        
        return prompt


#function to get implicit entities from the given attributes
def add_unique_attributes(df, attributes):
    if not attributes:
        False, {}, ''
    qry = ' and '.join(["{} == '{}'".format(k,v) for k,v in attributes.items() if k !='size' and k!='greeting'])
    if 'size' in attributes:
        if not qry:
            qry = 'size >='+ convertToNumber(attributes['size'])
        else:
            qry = qry + ' and size >='+ convertToNumber(attributes['size'])
    y = {}
    #print(qry)
    if not qry:
        return False, y
    subsetDF = df.query(qry)
    print(subsetDF)
    print('\n\n')
    if subsetDF.empty:
        return True, y,qry
    
    for col in subsetDF:
        if col == 'numberOfPeople':
            continue
        unique_vals = subsetDF[col].unique()
        if len(unique_vals) ==1 and col not in attributes:
            y[col] = unique_vals[0]
    return False, y,qry

def convertToNumber(numberword):
    if numberword =='one':
        return '1'
    elif numberword == 'two':
        return '2'
    elif numberword == 'three':
        return '3'
    elif numberword == 'four':
        return '4'
    elif numberword == 'five':
        return '5'
    elif numberword == 'six':
        return '6'
    elif numberword == 'seven':
        return '7'
    elif numberword == 'eight':
        return '8'
    elif numberword == 'nine':
        return '9'
    elif numberword == 'ten':
        return '10'
    
    


# In[ ]:

session = Session()

print ('BOT: Hi! How may I assist you?')

while True:
    
    inp = input('User: ')
    print ('BOT:', session.reply(inp))

