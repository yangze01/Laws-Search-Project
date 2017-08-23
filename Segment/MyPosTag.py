# coding=utf-8
from __future__ import print_function
from pyltp import *
import sys
import re
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]

class MyPostagger(object):
    def __init__(self):
        self.model = BasePath + '/ltp_data/pos.model'
        # self.lexicon = 'lexi.model'
        self.postagger = Postagger()
        self.postagger.load(self.model)

    def load_default_model(self):#
        self.postagger.load(self.model)
    def get_pos(self, words_list):
        return self.postagger.postag(words_list)

    def words2pos(self, words_list, poslist = []):
        # print(words_list)
        uword = ' '.join(words_list).encode('utf8')
        # print(uword)
        uword = uword.split(' ')

        if poslist == []:
            return words_list
        else:
            pos_set = set(poslist)
            postags = self.get_pos(uword)
            return_words = [words_list[i] for i in range(len(words_list)) if postags[i] in pos_set]
            return return_words

    def close(self):

        self.postagger.release()
        print("close the postagger {}".format(self.postagger.release()))


