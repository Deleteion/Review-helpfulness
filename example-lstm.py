#加载库
# -*- coding: utf-8 -*-
import matplotlib as matplotlib
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adam

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import pickle


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(len(train_df), len(test_df))
train_df.head()



stops = set(stopwords.words('english'))


def preprocess(text):
    # input: 'Hello are you ok?'
    # output: ['Hello', 'are', 'you', 'ok', '?']
    text = str(text)
    text = text.lower()

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)  
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.split()


word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


vocabulary = []
word2id = {}
id2word = {}

for df in [train_df, test_df]:
    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        for column in ['question1', 'question2']:
            q2n = []
            for word in preprocess(row[column]):
                if word in stops or word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    word2id[word] = len(vocabulary) + 1
                    id2word[len(vocabulary) + 1] = word
                    vocabulary.append(word)
                    q2n.append(word2id[word])
                else:
                    q2n.append(word2id[word])

            df.at[i, column] = q2n

embedding_dim = 300
embeddings = np.random.randn(len(vocabulary) + 1, embedding_dim)
embeddings[0] = 0
for index, word in enumerate(vocabulary):
    embeddings[index] = word2vec.word_vec(word)

del word2vec
print(len(vocabulary))


maxlen = max(train_df.question1.map(lambda x: len(x)).max(),
             train_df.question2.map(lambda x: len(x)).max(),
             test_df.question1.map(lambda x: len(x)).max(),
             test_df.question2.map(lambda x: len(x)).max())

valid_size = 40000
train_size = len(train_df) - valid_size

X = train_df[['question1', 'question2']]
Y = train_df['is_duplicate']

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=valid_size)
X_train = {'left': X_train.question1.values, 'right': X_train.question2.values}
X_valid = {'left': X_valid.question1.values, 'right': X_valid.question2.values}
Y_train = np.expand_dims(Y_train.values, axis=-1)
Y_valid = np.expand_dims(Y_valid.values, axis=-1)


X_train['left'] = np.array(pad_sequences(X_train['left'], maxlen=maxlen))
X_train['right'] = np.array(pad_sequences(X_train['right'], maxlen=maxlen))
X_valid['left'] = np.array(pad_sequences(X_valid['left'], maxlen=maxlen))
X_valid['right'] = np.array(pad_sequences(X_valid['right'], maxlen=maxlen))

print(X_train['left'].shape, X_train['right'].shape)
print(X_valid['left'].shape, X_valid['right'].shape)
print(Y_train.shape, Y_valid.shape)




hidden_size = 128
# gradient_clipping_norm = 1.25
gradient_clipping_norm = 1.25
batch_size = 256
epochs = 20

def exponent_neg_manhattan_distance(args):
    left, right = args
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))

left_input = Input(shape=(None,), dtype='int32')
right_input = Input(shape=(None,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=maxlen, trainable=False)

embedded_left = embedding_layer(left_input)
embedded_right = embedding_layer(right_input)

shared_lstm = LSTM(hidden_size)

left_output = shared_lstm(embedded_left)
right_output = shared_lstm(embedded_right)

malstm_distance = Lambda(exponent_neg_manhattan_distance, output_shape=(1,))([left_output, right_output])

malstm = Model([left_input, right_input], malstm_distance)

optimizer = Adam(clipnorm=gradient_clipping_norm,lr=0.001)
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

history = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=epochs,
                     validation_data=([X_valid['left'], X_valid['right']], Y_valid))



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


malstm.save('malstm.h5')
with open('data.pkl', 'wb') as fw:
    pickle.dump({'word2id': word2id, 'id2word': id2word}, fw)