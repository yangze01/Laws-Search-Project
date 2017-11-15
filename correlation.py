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


if __name__ == "__main__":
    print(1)
    # 将数据保存为词袋模型
    with open(BasePath + "/seg_corpus/data_corpus0.json", 'rb') as fp:
        data = json.load(fp)
    print(data.keys())

    # 将数据库保存为词袋模型
    dictionary_path = BasePath + "/correlation_data/dict.pickle"
    dictionary = make_dictionary(data['content_wordlist'], dictionary_path)
    print("the dictionary len is : {}".format(len(dictionary)))
    id2token = {value: key for key, value in dictionary.token2id.items()}
    # print(id2token)
    print(data['content_wordlist'])





