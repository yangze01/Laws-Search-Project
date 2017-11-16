#coding=utf8
from __future__ import print_function
from __future__ import division
from gensim import corpora, models, similarities
import os
import codecs
import datetime
import collections
import numpy as np
import pickle

import gc
from Segment.MySegment import *
from Segment.MyPosTag import *
import math
import json

def make_dictionary(seg_corpus, dictionary_file):
    '''
    :param seg_corpus: 分好词的文本数据
    :return: 建立的词典
    '''
    # dictionary_file = BasePath + "/data/zhapian_sen_word_dict.pickle"
    print(dictionary_path)
    if os.path.exists(dictionary_file):
        dictionary = corpora.Dictionary.load(dictionary_file)
    else:
        print("do not have a previous dict")
        dictionary = corpora.Dictionary(seg_corpus)
        print(dictionary_file)
        corpora.Dictionary.save(dictionary, dictionary_file)
    print("make dictionary finished")
    return dictionary

def doc2bag_corpus(dictionary, seg_corpus):

    bag_of_words_corpus = [dictionary.doc2bow(pdoc) for pdoc in seg_corpus]
    print("bag_of words finished finished")
    return bag_of_words_corpus

def save_json(save_path, data):
    with open(save_path, 'w+') as f:
        json.dump(data, f, ensure_ascii = False)
    print("json data saved as {}".format(save_path))



if __name__ == "__main__":
    print(1)

    # 将数据库保存为词袋模型
    with open(BasePath + "/seg_corpus/data_corpus0.json", 'rb') as fp:
        data = json.load(fp)
    print(data.keys())
    corpus = data['content_wordlist']
    dictionary_path = BasePath + "/correlation_data/dict.pickle"
    dictionary = make_dictionary(corpus, dictionary_path)
    print("the dictionary len is : {}".format(len(dictionary)))
    # id2token = {value: key for key, value in dictionary.token2id.items()}
    # print(id2token)
    print(' '.join(data['content_wordlist'][0]))
    print("the len of dictionary: {}".format(len(dictionary.token2id)))
    bag_corpus = doc2bag_corpus(dictionary, corpus)
    bag_sen_dict = dict()
    for i in range(0, len(bag_corpus)):
        print(i)
        tmp_sen = bag_corpus[i]
        bag_sen = [word_tuple[0] for word_tuple in tmp_sen]
        # print(tmp_sen)
        # print(bag_sen)
        bag_sen_dict[i+1] = bag_sen

    print("the len of bag_sen_dict is : {}".format(len(bag_sen_dict)))





























