import json
from pprint import pprint
import numpy as np
import re
import io
import nltk
import os
from keras.preprocessing.sequence import pad_sequences



def tokenize(sent):
	'''Return the tokens of a sentence including punctuation.
	Input: Sentence
	Output: A list of words. 
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	'''
	return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]


def tokenizeVal(sent):
	'''Return the tokens of a sentence including punctuation.
	Input: Sentence
	Output: A list of words + A list of starting index of each word in the sentence

	>>> tokenizeVal('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	[0, 4, 12, 16, 21, 23, 29, 32, 36, 41]
	'''
	tokenizedSent = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sent)]
	tokenIdx2CharIdx = [None] * len(tokenizedSent)
	idx = 0
	token_idx = 0
	while idx < len(sent) and token_idx < len(tokenizedSent):
		word = tokenizedSent[token_idx]
		if sent[idx:idx+len(word)] == word:
			tokenIdx2CharIdx[token_idx] = idx
			idx += len(word)
			token_idx += 1
		else:
			idx += 1
	return tokenizedSent, tokenIdx2CharIdx


def splitTrainDatasets(f):
	'''Input: jsonfile
		Output: Tokenized Sentences of Context, Question, a list of QuestionID,StartIdxAnswer, EndIndxAnswer, maxLenContext, maxLenQuestion'''

	tContext = []  # list of contexts paragraphs
	tQuestion = []  # list of questions
	tQuestion_id = []  # list of question id
	tStartIdxAnswer = []  # list of indices of the beginning word in each answer span
	tEndIdxAnswer = []  # list of indices of the ending word in each answer span
	tAnswerText = []  # list of the answer text
	maxLenContext = 0
	maxLenQuestion = 0
	for document in f['data']:
		# interate through each document
		paragraphs = document['paragraphs']

		for paragraph in paragraphs:
			context = paragraph['context']
			context1 = context.replace("''", '" ')
			context1 = context1.replace("``", '" ')
			contextTokenized = tokenize(context.lower())
			contextLength = len(contextTokenized)
			if contextLength > maxLenContext:
				maxLenContext = contextLength
			qas = paragraph['qas']
			for qa in qas:
				question = qa['question']
				question = question.replace("''", '" ')
				question = question.replace("``", '" ')
				questionTokenized = tokenize(question.lower())
				if len(questionTokenized) > maxLenQuestion:
					maxLenQuestion = len(questionTokenized)
				question_id = qa['id']
				answers = qa['answers']
				for answer in answers:
					answerText = answer['text']
					answerTokenized = tokenize(answerText.lower())
					# find indices of beginning/ending words of answer span
					# among tokenized context
					contextToAnswerFirstWord = context1[:answer['answer_start'] + len(answerTokenized[0])]
					answerBeginIndex = len(tokenize(contextToAnswerFirstWord.lower())) - 1
					answerEndIndex = answerBeginIndex + len(answerTokenized) - 1

					tContext.append(contextTokenized)
					tQuestion.append(questionTokenized)
					tQuestion_id.append(str(question_id))
					tStartIdxAnswer.append(answerBeginIndex)
					tEndIdxAnswer.append(answerEndIndex)
					tAnswerText.append(answerText)

	return tContext, tQuestion, tQuestion_id, tStartIdxAnswer, tEndIdxAnswer, tAnswerText, maxLenContext, maxLenQuestion



def splitValDatasets(f):
	'''Given a parsed Json data object, split the object into training context (paragraph), question, answer matrices, 
	   and keep track of max context and question lengths.
	'''
	vContext = []  # list of tokenized contexts paragraphs  [['super', 'bowl', '50', 'was', 'an', 'american', 'football',..]]
	vQuestion = []  # list of questions
	vQuestion_id = []  # list of question id
	vToken2CharIdx = [] # first index of every word in a sentence
	vContextOriginal = [] # sentences of the paragraphs ['Super Bowl 50 was an American football]
	maxLenContext = 0
	maxLenQuestion = 0
	for document in f['data']:
		# interate through each document
		paragraphs = document['paragraphs']

		for paragraph in paragraphs:
			context = paragraph['context']
			context1 = context.replace("''", '" ')
			context1 = context1.replace("``", '" ')
			contextTokenized, tokenIdx2CharIdx = tokenizeVal(context1.lower())
			contextLength = len(contextTokenized)
			if contextLength > maxLenContext:
				maxLenContext = contextLength
			qas = paragraph['qas']
			for qa in qas:
				question = qa['question']
				question = question.replace("''", '" ')
				question = question.replace("``", '" ')
				questionTokenized = tokenize(question.lower())
				if len(questionTokenized) > maxLenQuestion:
					maxLenQuestion = len(questionTokenized)
				question_id = qa['id']
				answers = qa['answers']

				vToken2CharIdx.append(tokenIdx2CharIdx)
				vContextOriginal.append(context)
				vContext.append(contextTokenized)
				vQuestion.append(questionTokenized)
				vQuestion_id.append(str(question_id))

	return vContext, vToken2CharIdx, vContextOriginal, vQuestion, vQuestion_id, maxLenContext, maxLenQuestion


def vectorizeTrainData(xContext, xQuestion, xAnswerBeing, xAnswerEnd, word_index, context_maxlen, question_maxlen):
	'''Vectorize the words to their respective index and pad context to max context length and question to max question length.
	   Answers vectors are padded to the max context length as well.
	'''
	X = []
	Xq = []
	YBegin = []
	YEnd = []
	for i in range(len(xContext)):
		x = [word_index[w] for w in xContext[i]]
		xq = [word_index[w] for w in xQuestion[i]]
		# map the first and last words of answer span to one-hot
		# representations
		y_Begin = np.zeros(len(xContext[i]))
		y_Begin[xAnswerBeing[i]] = 1
		y_End = np.zeros(len(xContext[i]))
		y_End[xAnswerEnd[i]] = 1
		X.append(x)
		Xq.append(xq)
		YBegin.append(y_Begin)
		YEnd.append(y_End)
	return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post'), pad_sequences(YBegin, maxlen=context_maxlen, padding='post'), pad_sequences(YEnd, maxlen=context_maxlen, padding='post')

# for validation dataset


def vectorizeValData(xContext, xQuestion, word_index, context_maxlen, question_maxlen):
	'''Vectorize the words to their respective index and pad context to max context length and question to max question length.
	   Answers vectors are padded to the max context length as well.
	'''
	X = []
	Xq = []
	YBegin = []
	YEnd = []
	for i in range(len(xContext)):
		x = [word_index[w] for w in xContext[i]]
		xq = [word_index[w] for w in xQuestion[i]]

		X.append(x)
		Xq.append(xq)

	return pad_sequences(X, maxlen=context_maxlen, padding='post'), pad_sequences(Xq, maxlen=question_maxlen, padding='post')






