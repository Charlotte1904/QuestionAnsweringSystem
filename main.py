import json
import numpy as np
import re
import io
import nltk
import os
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Merge, Dropout, RepeatVector, Activation, merge, Lambda, Flatten, Reshape
from keras.layers import LSTM, Bidirectional, TimeDistributed, GRU, AveragePooling1D, Reshape, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from simpleAttention import Attention, SimpleAttention, SSimpleAttention
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocessing_data import * 
from pprint import pprint



# nltk.download() # after downloaded nltk, can cancel out this line 
# when asked which identifier - type punkt

print('Loading in Training data...')
with io.open('train-v1.1.json', 'r', encoding='utf-8') as f:
    raw_data = f.read()
trainData = json.loads(raw_data)

tContext, tQuestion, tQuestion_id, tAnswerBegin, tAnswerEnd, tAnswerText, maxLenTContext, maxLenTQuestion = splitTrainDatasets(trainData)

print('Loading in Validation data...')

with io.open('dev-v1.1.json', 'r', encoding='utf-8') as f:
    raw_val = f.read()
valData = json.loads(raw_val)

vContext, vToken2CharIdx, vContextSentences, vQuestion, vQuestion_id, maxLenVContext, maxLenVQuestion = splitValDatasets(valData)

print('Building vocabulary...')

vocab = {}
for setences in tContext + tQuestion + vContext + vQuestion:
    for word in setences:
        if word not in vocab:
            vocab[word] = 1

vocab = sorted(vocab.keys())
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_index = dict((c, i + 1) for i, c in enumerate(vocab))
context_maxlen = max(maxLenTContext, maxLenVContext)
question_maxlen = max(maxLenTQuestion, maxLenVQuestion)



print('Begin vectoring process...')
# tX: training Context, tXq: training Question, tYBegin: training Answer
# Begin ptr, tYEnd: training Answer End ptr


tX, tXq, tYBegin, tYEnd = vectorizeTrainData(tContext, tQuestion, tAnswerBegin, tAnswerEnd, word_index, context_maxlen, question_maxlen)
# shuffle train data
randindex = np.random.permutation(tX.shape[0])
tX = tX[randindex, :]
tXq = tXq[randindex, :]
tYBegin = tYBegin[randindex, :]
tYEnd = tYEnd[randindex, :]
# vX: validation Context, vXq: validation Question
vX, vXq = vectorizeValData(vContext, vQuestion, word_index, context_maxlen, question_maxlen)
print('Vectoring process completed.')

# print('tX.shape = {}'.format(tX.shape))
# print('tXq.shape = {}'.format(tXq.shape))
# print('tYBegin.shape = {}'.format(tYBegin.shape))
# print('tYEnd.shape = {}'.format(tYEnd.shape))
# print('vX.shape = {}'.format(vX.shape))
# print('vXq.shape = {}'.format(vXq.shape))
# print('context_maxlen, question_maxlen = {}, {}'.format(
#     context_maxlen, question_maxlen))

print('Preparing embedding matrix.')

# prepare embedding matrix
nb_words = len(word_index)
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = context_maxlen

embeddings_index = {}

f = open(os.path.join('glove.6B', 'glove.6B.100d.txt'))
# f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Defining Model...")

question_input = Input(shape=(question_maxlen,),
                       dtype='int32', name='question_input')
context_input = Input(shape=(context_maxlen,),
                      dtype='int32', name='context_input')
questionEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                         mask_zero=True, weights=[embedding_matrix],
                         input_length=question_maxlen, trainable=False)(question_input)
contextEmbd = Embedding(output_dim=EMBEDDING_DIM, input_dim=vocab_size,
                        mask_zero=True, weights=[embedding_matrix],
                        input_length=context_maxlen, trainable=False)(context_input)

Q = Bidirectional(LSTM(64, return_sequences=True))(questionEmbd)
D = Bidirectional(LSTM(64, return_sequences=True))(contextEmbd)
# Q1 = Bidirectional(LSTM(96, return_sequences=True))(Q)
# D1 = Bidirectional(LSTM(96, return_sequences=True))(D)

Q2 = LSTM(128, return_sequences=False)(Q)
D2 = LSTM(128, return_sequences=False)(D)
Q3 = SimpleAttention(128, D2, 128, return_sequences=True)(Q)
D3 = SimpleAttention(128, Q2, 128, return_sequences=True)(D)
Q4 = LSTM(128, return_sequences=False)(Q3)
D4 = LSTM(128, return_sequences=False)(D3)
L = merge([D4, Q4], mode='concat')

answerPtrBegin_output = Dense(context_maxlen, activation='softmax')(L)
Lmerge = merge([L, answerPtrBegin_output], mode='concat', name='merge2')
answerPtrEnd_output = Dense(context_maxlen, activation='softmax')(Lmerge)


model = Model(input=[context_input, question_input], output=[
              answerPtrBegin_output, answerPtrEnd_output])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[.04, 0.04], metrics=['accuracy'])

model.summary()

print('ModelCheckpoint')
checkpoint1 = EarlyStopping(monitor='val_dense_1_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')
checkpoint2 = EarlyStopping(monitor='val_dense_2_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')
callbacks_list = [checkpoint1, checkpoint2]  

model.fit([tX, tXq], [tYBegin, tYEnd], epochs=5, batch_size=256, shuffle=True, validation_split=0.2,
          callbacks=callbacks_list)

print('Finish Training...')


# model.save('model1')
# D.save('D.h5')
print('-------------------------')
print('Start Predicting Process... ')

predictions = model.predict([vX, vXq], batch_size=128)
ansBegin = np.zeros((predictions[0].shape[0],), dtype=np.int32)
ansEnd = np.zeros((predictions[1].shape[0],), dtype=np.int32)
for i in range(predictions[0].shape[0]):
    ansBegin[i] = predictions[0][i, :].argmax()
    ansEnd[i] = predictions[1][i, :].argmax()


answers = {}
for i in range(len(vQuestion_id)):
    # print i
    if ansBegin[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = ""
    elif ansEnd[i] >= len(vContext[i]):
        answers[vQuestion_id[i]] = vContextSentences[i][vToken2CharIdx[i][ansBegin[i]]:]
    else:
        answers[vQuestion_id[i]] = vContextSentences[i][vToken2CharIdx[i][ansBegin[i]]:vToken2CharIdx[i][ansEnd[i]]+len(vContext[i][ansEnd[i]])]
        # The original context paragraph with full sentences of 'the ith sample' 
        # [the first answer idex word within vToken2CharId in'the ith sample': the last answer index word + the len of the last word]





print('Writing out Predictions...')

# write out answers to json file
with io.open('FiveEpo-Prediction-newmodel.json', 'w', encoding='utf-8') as f:
    f.write((json.dumps(answers, ensure_ascii=False)))

