
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("imdb_master.csv",encoding="utf-8")

df=df[df["label"]!="unsup"]
# In[3]:


df=df.replace("neg", 0)
df=df.replace("pos", 1)
df.head()

df['review'] = df['review'].str.replace("[^a-zA-Z#]", " ")

# In[4]:

traindf=df[df["type"]=="train"].sample(frac=0.5,random_state=0)
testdf=df[df["type"]=="test"].sample(frac=0.2,random_state=0)
traindf=traindf[["review","label"]]
testdf=testdf[["review","label"]]
print(len(traindf))
print(len(testdf))


# In[5]:

import nltk
from nltk.tokenize import word_tokenize
X_train=[word_tokenize(i) for i in traindf["review"].values]
y_train=traindf["label"].values
X_test=[word_tokenize(i) for i in testdf["review"].values]
y_test=testdf["label"].values

print(X_train[0])
# In[6]:


words = set([])
for s in X_train:
    for w in s:
        words.add(w.lower())
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs


# In[7]:


import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import multiprocessing
import os


# In[11]:


#Declare Model Parameters
cbow = 0
skipgram = 1
EMB_DIM = 150 #more dimensions, more computationally expensive to train
min_word_count = 1
workers = multiprocessing.cpu_count() #based on computer cpu count
context_size = 7
downsampling = 1e-3
learning_rate = 0.025 #initial learning rate
min_learning_rate = 0.025 #fixated learning rate
num_epoch = 10


# In[12]:


w2v = Word2Vec(
    sg = skipgram,
    hs = 1, #hierarchical softmax
    size = EMB_DIM,
    min_count = min_word_count, 
    workers = workers,
    window = context_size, 
    sample = downsampling, 
    alpha = learning_rate, 
    min_alpha = min_learning_rate
)


# In[13]:


#w2v.build_vocab(X_train)
#w2v.train(X_train,epochs=num_epoch,total_examples=w2v.corpus_count)
#words = list(w2v.wv.vocab)
#print('Vocabulary size: %d' % len(words))
#save model in ASCII (word2vec) format
#filename = 'embedding_word2vec.txt'
#w2v.wv.save_word2vec_format(filename, binary=False)

embeddings_index={}
f=open(os.path.join('','embedding_word2vec.txt '),encoding="utf-8")
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()


# In[ ]:


train_sentences_X, test_sentences_X = [], []

num_words=len(word2index)+1
embedding_matrix=np.zeros((num_words,EMB_DIM))
print(word2index)
for word,i in word2index.items():
    if i>num_words:
        continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector

for s in X_train:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    train_sentences_X.append(s_int)
    
for s in X_test:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])
 
    test_sentences_X.append(s_int)


MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)  # 271

from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
# In[ ]:

from keras import Sequential
from keras.initializers import Constant
from keras.layers import Embedding, LSTM, Dense, Dropout,CuDNNLSTM,Bidirectional,TimeDistributed,Activation,Flatten
from keras.utils import plot_model
model=Sequential() 
embedding_layer=Embedding(num_words,EMB_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=MAX_LENGTH,trainable=True)
model.add(embedding_layer)
model.add(Bidirectional(CuDNNLSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy'])
print(model.summary())

history=model.fit(train_sentences_X, y_train, validation_split=0.2, batch_size=64, epochs=10)

scores = model.evaluate(test_sentences_X, y_test, verbose=0)
print('Test accuracy:', scores[1])
model.save("model.h5")
plot_model(model, to_file='model.png')
import matplotlib.pyplot as plt
# Plot training & validation accuracy values
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("fake_acc.png")

plt.figure( )
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("loss.png")