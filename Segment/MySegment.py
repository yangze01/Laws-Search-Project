# coding=utf-8
from __future__ import print_function
from pyltp import *
import os
import sys
import re
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]


# reminder = open(BasePath + '/data/law_left.txt')
# s = reminder.read()
# reminder_word_from_file = s.split()

# f = open(BasePath + "/data/law_left.txt")
# from Segment.MyPosTag import  *
# word_set = f.read()
# word_set = word_set.split()
# pattern_str = ''.join([i + "|" for i in word_set])

# print(pattern_str[:-1])

def is_utf8(s):
	try:
		s.decode('utf-8')
		return True
	except UnicodeError:
		return False

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass
	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass
	return False

def word_reminder(line):
    with open(BasePath + '/data/law_left.txt') as reminder:
        s = reminder.read()
        reminder_word_from_file = s.split()
    reminder_word_from_file = set(reminder_word_from_file)
    total_reminder_words_set = reminder_word_from_file
    tokens = [w for w in line if w in total_reminder_words_set]
    # print("finish one reminder")
    return tokens

def stopword_remover(line):
    with open(BasePath + '/data/stop_word.txt', 'r') as stops:
        s = stops.read()
        stop_words_from_file = s.split()
    stop_words_from_file = set(stop_words_from_file)
    total_stop_words_set = stop_words_from_file

    tokens = [w for w in line if not (w in total_stop_words_set or is_number(w) or ((not is_ascii(w)) and len(w) < 4))]
    # print(' '.join(tokens))
    # if len(tokens) > 3:
        # print(' '.join([t for t in tokens]), file=outFile)
    return tokens

class MySegment(object):
    def __init__(self):
        self.model = BasePath + '/ltp_data/cws.model'
        self.lexicon = 'lexi.model'

        self.segmentor = Segmentor()
        # self.segmentor.load(self.model)
        self.segmentor.load_with_lexicon(self.model, BasePath + '/Segment/law_dict.txt')  # 加载模型，第二个参数是您的外部词典文件路径


    def load_default_model(self):
        self.segmentor.load(self.model)

    def paraph2word(self, paraph):
        sen_list = self.paraph2sen(paraph)
        word_list = self.senlist2word(sen_list)
        return word_list

    def sen2word(self, sen):
        word_obj = self.segmentor.segment(sen)
        word_list = list(word_obj)
        word_list = stopword_remover(word_list)
        return word_list


    # def sen2word(self, sen):
    #     pattern = re.compile(pattern_str[:-1].decode("utf8"))
    #     matcher1 = re.search(pattern, sen.decode('utf8'))
    #     # nonsentence = sen
    #     if(matcher1):
    #         word_obj = self.segmentor.segment(sen)
    #         word_list = list(word_obj)
    #         word_list = stopword_remover(word_list)
    #         return word_list
    #     else:
    #         return []


    def senlist2word(self,sentence_list):
        # print(pattern_str[:-1])
        '''
            input:
                sentent the sen to be seg
            output:
                the word list
        '''
        word_list = list()
        for sentence in sentence_list:
            # print(sentence)
            # sentence.find()
            word_obj = self.sen2word(sentence)
            word_list = word_list + word_obj
        return word_list

    def paraph2sen(self,paraph):
        sentence_obj = SentenceSplitter.split(paraph.encode('utf8'))
        sentence_list = list(sentence_obj)
        return list(sentence_list)

    def remove_punctuation(self,sentence):
        # m=sentence.replace(string.punctuation,"")
        # return m
        return ''.join(re.findall(u'[\u4e00-\u9fff]+', sentence)).encode('utf8')

    def close(self):
        self.segmentor.release()
        # print("close the segmentor {}".format(self.segmentor.release()))

if __name__ == "__main__":
    print(1)
    # file_path = BasePath + "/data/stop_word.txt"
    #
    myseg = MySegment()
    # sen_list = myseg.paraph2sen(new_tuple[2])
    # wordlist = myseg.senlist2word(sen_list)
    #
    # print(','.join(wordlist))
    # string.split(',')
