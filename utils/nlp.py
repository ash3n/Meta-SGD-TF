from .core import *
import numpy as np
import os

glove = None

def load_glove(gloveFile=C_.workspace+'/data/glove.6B/glove.6B.50d.txt'):
    f = open(gloveFile, 'r', encoding='utf-8')
    global glove
    glove = class_like(dict, 'glove')()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        glove[word] = embedding
    glove.indices = {word:index for index, word in enumerate(glove)}
    glove.embeddings = [glove[word] for word in glove]

def get_glove():
    if glove is None:
        print('Loading GloVe..')
        load_glove()
        print('GloVe successfully loaded!')
    return glove

def batch_tokenize(sentences, verbose=True, delimiter='thisisadelimiter'):
    sentence_batch = ''
    if verbose: print('Batching sentences..')
    for sentence in sentences:
        sentence_batch += '%s %s '%(sentence,delimiter)
    if verbose: print('Tokenizing batch..')
    tokenized_batch = tokenize(sentence_batch)
    if verbose: print('Splitting batch..')
    tokenized = []
    tokens = []
    for token in tokenized_batch:
        if token == delimiter:
            tokenized.append(tokens)
            tokens = []
        else: tokens.append(token)
    if verbose: print('Successfully tokenized!')
    return tokenized

def glove_index(tokenized, verbose=True, delimiter='thisisadelimiter'):
    glove = get_glove()
    tokenized = [tokenized] if type(tokenized[0]) != list else tokenized
    indexed = [[glove.indices[token] if token in glove.indices else len(glove) 
        for token in tokens] for tokens in tokenized]
    if verbose: print('Successfully indexed!')
    indexed = indexed[0] if len(indexed) == 1 else indexed
    return indexed

def tokenize(string): # requires workspace/corenlp
    input_file = open(C_.workspace+'/data/input.txt', 'w+')
    input_file.write(printable(string))
    input_file.close()
    abshome = os.path.abspath(C_.workspace).replace('\\', '/')
    plug = (C_.homepath, C_.homepath, C_.homepath)
    tokenizer_cmd = 'java --add-modules java.se.ee -Xmx2g -cp "%s/corenlp/*" edu.stanford.nlp.process.PTBTokenizer %s/data/input.txt > %s/data/output.txt'%plug
    os.system(tokenizer_cmd)
    output_file = open(C_.workspace+'/data/output.txt', 'r')
    output = [v.rstrip() for v in output_file.readlines()]
    output_file.close()
    return output
    
def printable(string, replace='unknownchar'):
    return ''.join([i if ord(i) < 128 else replace for i in string])