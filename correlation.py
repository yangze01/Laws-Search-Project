#coding=utf8
from __future__ import print_function
from __future__ import division
from gensim import corpora, models, similarities
import os
import codecs
import datetime
import collections
import numpy as np
from get_element import *
import pickle
from data_helper import *
import gc
from Segment.MySegment import *
from Segment.MyPosTag import *
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from __future__ import division
from itertools import combinations
class Vocab:

    def __init__(self, token2index=None, index2token = None):
        self._token2index = token2index or {}
        self._index2token = index2token or []

    def feed(self, token):
        if token not in self._token2index:
            # allocate new index for this token
            index = len(self._token2index)
            self._token2index[token] = index
            self._index2token.append(token)

        return self._token2index[token]

    @property
    def size(self):
        return len(self._token2index)

    def token(self, index):
        return self._index2token[index]

    def __getitem__(self, token):
        index = self.get(token)
        if index is None:
            raise KeyError(token)
        return index

    def get(self, token, default=None):
        return self._token2index.get(token, default)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self._token2index, self._index2token), f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            token2index, index2token = pickle.load(f)
        # print(token2index)
        return cls(token2index, index2token)

def make_dictionary(seg_corpus):
    '''
    :param seg_corpus: 分好词的文本数据
    :return: 建立的词典
    '''
    dictionary_file = BasePath + "/data/zhapian_sen_word_dict.pickle"
    if os.path.exists(dictionary_file):
        dictionary = corpora.Dictionary.load(dictionary_file)
    else:
        print("do not have a previous dict")
        dictionary = corpora.Dictionary(seg_corpus)
        corpora.Dictionary.save(dictionary, dictionary_file)
    print("make dictionary finished")
    return dictionary

def doc2bag_corpus(dictionary, seg_corpus):

    bag_of_words_corpus = [dictionary.doc2bow(pdoc) for pdoc in seg_corpus]
    print("bag_of words finished finished")
    return bag_of_words_corpus

def save_seg_document(file_path, corpus_list):
    seg_dict = dict()
    filepath =file_path
    seg_dict["content"] = corpus_list
    encode_json = json.dumps(seg_dict, ensure_ascii = False)
    with open(filepath, "w+") as f:
        f.write(encode_json)
    print("seg document saved as {}".format(filepath))

def load_seg_document(file_path):
    with open(file_path,"r+") as f:
        jsondata = f.read()
        jsondict = json.loads(jsondata)
    content_list = jsondict['content']
    return content_list

def save_bag_corpus(bag_corpus_dict):
    # seg_dict = dict()
    filepath = BasePath + "/data/sen_bag_word_corpus" + ".txt"
    # seg_dict["content"] = corpus_list
    encode_json = json.dumps(bag_corpus_dict, ensure_ascii = False)
    with open(filepath, "w+") as f:
        f.write(encode_json)
    print("bag_corpus_dict saved as {}".format(filepath))

def load_bag_corpus():
    '''
        将词袋表示的句子重新加载
    :return:
    '''
    file_path = BasePath + "/data/sen_bag_word_corpus" + ".txt"
    with open(file_path,"r+") as f:
        jsondata = f.read()
        jsondict = json.loads(jsondata)
    # content_list = jsondict['content']
    return jsondict




def save_corpus(is_select = False):
    '''
        将语料分割为以句子为单位的行
    :return: 将数据保存为zhapian_sen_word_corpus
    '''
    # word_vocab = Vocab()
    # word_vocab.feed('|')  # <unk> is at index 0 in word vocab
    myseg = MySegment()
    mypos = MyPostagger()
    opt = OptOnMysql()
    opt_on_documents = DocumentsOnMysql()
    seg_corpus = list()
    it_iter = opt_on_documents.findbycriminal("诈骗罪")
    # print(it[0][0])
    for it in it_iter:
        # print(it[25])
        print(it[0])
        content = it[25]
        senlist = myseg.paraph2sen(content.replace('|', ''))
        if is_select == False:
            for sentence in senlist:
                word_set = set(myseg.sen2word(sentence.encode('utf8')))
                word_list = list(word_set)
                seg_corpus.append(word_list)
        else:
            for sentence in senlist:
                word_set = myseg.sen2word(sentence.encode('utf8'))
                word_list = mypos.words2pos(word_set, ['n', 'nl', 'ns', 'v'])
                seg_corpus.append(list(set(word_list)))

    # for id in range(1, 11152):
    #     it = opt_on_documents.getById(id)
    #     content_id = it[0]
    #     content = it[5]
    #     print(content_id)
    #     senlist = myseg.paraph2sen(content.replace('|', ''))
    #     print("the length of senlist is : {} ".format(len(senlist)))#
    #     for sentence in senlist:
    #         word_set = set(myseg.sen2word(sentence.encode('utf8')))
    #         word_list = list(word_set)
    #         seg_corpus.append(word_list)
    filepath = BasePath + "/data/zhapian_sen_word_corpus_pos_select" + ".txt"
    save_seg_document(filepath, seg_corpus)
    opt.connClose()
    opt_on_documents.connClose()
    myseg.close()
    mypos.close()

def nmi_get(cij, ci, cj, count_sen):
    # nmi = nmi_get([value, l_10, l_01], w_1_1, w_2_1, sen_size)
    nmi_list = []
    cij.append(count_sen - cij[0] - cij[1] - cij[2])
    pij = np.array(cij) / count_sen
    pi = ci / count_sen
    pj = cj / count_sen
    for i in range(0, 4):
        if (pij[i] == 0):
            nmi_list.append(0)
            continue
        if i == 0:
            nmi_list.append(pij[i]*np.log2(pij[i]/(pi*pj)))
        elif i == 1:
            nmi_list.append(pij[i] * np.log2(pij[i] / (pi * (1 - pj))))
        elif i == 2:
            nmi_list.append(pij[i] * np.log2(pij[i] / ((1 - pi) * pj)))
        elif i == 3:
            nmi_list.append(pij[i] * np.log2(pij[i] / ((1 - pi) * (1 - pj))))

    return np.sum(np.array(nmi_list))

if __name__ == "__main__":
   #  save_corpus(is_select=True)
   #
   #  # 将数据保存为词袋模型
   #  zhapian_sen_word_corpus = load_seg_document(BasePath + "/data/zhapian_sen_word_corpus_pos_select" + ".txt")
   #  dictionary = make_dictionary(zhapian_sen_word_corpus)
   #  print("the dictionary len is : {}".format(len(dictionary)))
   #  # print(dictionary.id2token)
   #  id2token = {value: key for key, value in dictionary.token2id.items()}
   #
   #  print("the len of dictionary: {}".format(len(dictionary.token2id)))
   #  bag_corpus = doc2bag_corpus(dictionary, zhapian_sen_word_corpus)
   #  print("the len of bag_corpus is : {}".format(len(bag_corpus)))
   #  bag_sen_dict = dict()
   #  for i in range(0, len(bag_corpus)):
   #      print(i)
   #      tmp_sen = bag_corpus[i]
   #      bag_sen = [word_tuple[0] for word_tuple in tmp_sen]
   #      # print(tmp_sen)
   #      # print(bag_sen)
   #      bag_sen_dict[i+1] = bag_sen
   #  print("the len of bag_sen_dict is : {}".format(len(bag_sen_dict)))
   #  save_bag_corpus(bag_sen_dict)
   #
   #
   #
   # # # 统计每个单词出现的句子
   #  count_word_in_sen_dict = dict()
   #  bag_sen_dict = load_bag_corpus() #从0开始
   #  print(len(bag_sen_dict))
   #
   #  for i in range(1, len(bag_sen_dict)+1):
   #      # print("+++++++++++++++++++++++++++")
   #      # print(i)
   #      # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
   #      tmp_word_id_list = bag_sen_dict[str(i)]
   #      for word_id in tmp_word_id_list:
   #          # print(word_id)
   #          if(word_id == 0):
   #              print(tmp_word_id_list)
   #          if word_id not in count_word_in_sen_dict:
   #              count_word_in_sen_dict[word_id] = [i]
   #          else:
   #              count_word_in_sen_dict[word_id].append(i)
   #
   #  count_word_sen_filepath = BasePath + "/data/zhapian_count_word_in_sen_list.txt"
   #  encode_json = json.dumps(count_word_in_sen_dict, ensure_ascii=False)
   #  with open(count_word_sen_filepath, "w+") as f:
   #      f.write(encode_json)
   #  print("seg count_word_sen_filepath saved as {}".format(count_word_sen_filepath))
   #  del encode_json
   #  del count_word_in_sen_dict
   #  gc.collect()
    """
    # 统计词对出现的次数
    # count_word_sen_filepath = BasePath + "/data/zhapian_count_word_in_sen_list.txt"
    # with open(count_word_sen_filepath,"r+") as f:
    #     jsondata = f.read()
    #     word_in_sen_dict = json.loads(jsondata)
    # print("the len of worddict is: {}".format(len(word_in_sen_dict)))
    #
    # word_pair = dict()
    #
    # # for id1 in range(0, len(word_in_sen_dict)):
    # #     for id2 in range(id1+1, len(word_in_sen_dict)):
    # for id1 in range(0, 1000):
    #     for id2 in range(id1+1, 1000):
    #         print(id1, id2)
    #         l1 = set(word_in_sen_dict[str(id1)]) & set(word_in_sen_dict[str(id2)])
    #         # print(len(l1))
    #         if(len(l1) < 1):
    #             continue
    #         l2 = set(word_in_sen_dict[str(id1)]) - set(word_in_sen_dict[str(id2)])
    #         l3 = set(word_in_sen_dict[str(id2)]) - set(word_in_sen_dict[str(id1)])
    #         word_pair[(id1, id2)] = [len(l1), len(l2), len(l3)]
    #
    # # 保存词对统计
    # countnum_word_in_sen_filepath = BasePath + "/data/countnum_word_in_sen.txt"
    # count_word_sen_filepath = BasePath + "/data/zhapian_count_word_in_sen_list.txt"
    # encode_json = json.dumps(word_pair, ensure_ascii=False)
    # with open(countnum_word_in_sen_filepath, "w+") as f:
    #  f.write(encode_json)
    # print("seg countnum_word_in_sen_filepath saved as {}".format(count_word_sen_filepath))
    """
############################################################################################
    # 统计候选词对
    # count_word_in_sen_dict = dict()
    # bag_sen_dict = load_bag_corpus()  # 从0开始
    # print(len(bag_sen_dict))

    # candidate_dict = dict()
    # for i in range(1, len(bag_sen_dict) + 1):
    #     print(i)
    #     comb = list(combinations(bag_sen_dict[str(i)],2))
    #
    #     for com_tuple in comb:
    #         if str(com_tuple) not in candidate_dict:
    #             candidate_dict[str(com_tuple)] = 1
    #         elif(str(com_tuple) in candidate_dict):
    #             candidate_dict[str(com_tuple)] += 1
    #             # print(candidate_dict[com_tuple])
    # print("the len of candidate is : {}".format(len(candidate_dict))) #10735325
    #
    # # 保存候选词对
    # candidate_file_path = BasePath + "/data/zhapian_candidate_word_pair.txt"
    # encode_json = json.dumps(candidate_dict, ensure_ascii=False)
    # with open(candidate_file_path, "w+") as f:
    #     f.write(encode_json)
    # print("seg countnum_word_in_sen_filepath saved as {}".format(candidate_file_path))
    #
    #
    # # 挑选高频词对，小于2的抛弃
    # candidate_file_path = BasePath + "/data/zhapian_candidate_word_pair.txt"
    # with open(candidate_file_path, "r+") as f:
    #     jsondata = f.read()
    #     candidate_high_frequence_dict = json.loads(jsondata)
    # print("the len of candidate_word_pair is: {}".format(len(candidate_high_frequence_dict)))
    # i = 1
    # for key,value in candidate_high_frequence_dict.items():
    #     print(i)
    #     i += 1
    #     # if(value>=2):
    #     #     print(i)
    #     #     i = i + 1
    #     #     print(value)
    #     if value < 10:
    #         del candidate_high_frequence_dict[key]
    #         # candidate_high_frequence_dict
    #
    # # 保存高频词对
    # candidate_high_freqence_file_path = BasePath + "/data/high_frequence_word_pair_10.txt"
    #
    # encode_json = json.dumps(candidate_high_frequence_dict, ensure_ascii=False)
    # with open(candidate_high_freqence_file_path, "w+") as f:
    #     f.write(encode_json)
    # print("high_frequence_word_pair saved as {}".format(candidate_high_freqence_file_path))
    # print("the len of the high_frequence word pair is : {}".format(len(candidate_high_frequence_dict)))
    #
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # 对高频词对统计11 10 01
    # 加载高频词对 (id1, id2) = count(11)
    candidate_high_freqence_file_path = BasePath + "/data/high_frequence_word_pair_10.txt"
    with open(candidate_high_freqence_file_path, "r+") as f:
        jsondata = f.read()
        candidate_high_frequence_dict = json.loads(jsondata)
        # print(candidate_high_frequence_dict)
    print("the len of candidate_high_frequence_dict is: {}".format(len(candidate_high_frequence_dict)))
    # 加载词---句子集合 w_id = [s1, s2, s3, s4 ...]
    count_word_sen_filepath = BasePath + "/data/zhapian_count_word_in_sen_list.txt"
    with open(count_word_sen_filepath, "r+") as f:
        jsondata = f.read()
        word_in_sen_dict = json.loads(jsondata)
    print("the len of word_in_sen_dict is: {}".format(len(word_in_sen_dict)))
    # 统计高频词的相关统计量，并转换为 (w1, w2) = [count(11), count(10), count(01)]
    bag_sen_dict = load_bag_corpus()
    sen_size = len(bag_sen_dict)#1951374#858277
    i = 1
    time_str = datetime.datetime.now().isoformat()
    print("the start time is : " + str(time_str))
    for key,value in candidate_high_frequence_dict.items():
        if(value == 0):
            print(key, value)

        if(i%1000 == 0):
            time_str = datetime.datetime.now().isoformat()
            print("the " + str(i) + " candidates use time is : " + str(time_str))
            print(i)

        i += 1
        id_list = key[1:-1].split(', ')
        w_1_1 = len(set(word_in_sen_dict[str(id_list[0])]))
        w_2_1 = len(set(word_in_sen_dict[str(id_list[1])]))

        l_10 = len(set(word_in_sen_dict[str(id_list[0])]) - set(word_in_sen_dict[str(id_list[1])]))
        l_01 = len(set(word_in_sen_dict[str(id_list[1])]) - set(word_in_sen_dict[str(id_list[0])]))

        nmi = nmi_get([value, l_10, l_01], w_1_1, w_2_1, sen_size)
        if(math.isnan(nmi)):
            print(value, l_10, l_01, sen_size - value - l_10 - l_01, w_1_1, w_2_1)
            # print(key, value, nmi)
            del candidate_high_frequence_dict[key]
            continue

        candidate_high_frequence_dict[key] = [value, nmi]
    # 保存高频词统计量
    candidate_high_freqence_count_file_path = BasePath + "/data/high_frequence_word_count_pair_10.txt"
    encode_json = json.dumps(candidate_high_frequence_dict, ensure_ascii=False)
    with open(candidate_high_freqence_count_file_path, "w+") as f:
        f.write(encode_json)
    print("high_frequence_word_pair saved as {}".format(candidate_high_freqence_count_file_path))

    candidate_high_freqence_count_file_path = BasePath + "/data/high_frequence_word_count_pair_10.txt"




    # test数据统计
    zhapian_sen_word_corpus = load_seg_document(BasePath + "/data/zhapian_sen_word_corpus" + ".txt")
    dictionary = make_dictionary(zhapian_sen_word_corpus)
    print("the len of dictionary is : " + str(len(dictionary)))
    print(dictionary[375])
    #
    id2token = {value: key for key, value in dictionary.items()}
    # print(id2token)

    # for key,value in dictionary.items():
        # print(key)
        # print(value)
    import csv
    csvfile1 = open(BasePath + '/csv_tuple_pos_select.csv', 'wb')
    writer1 = csv.writer(csvfile1)
    writer1.writerow(['w1','w2','count','pmi'])

    csvfile2 = open(BasePath + '/csv_dict_pos_select.csv', 'wb')
    writer2 = csv.writer(csvfile2)
    writer2.writerow(['token', 'id'])

    candidate_high_freqence_file_path = BasePath + "/data/high_frequence_word_count_pair_10.txt"
    with open(candidate_high_freqence_file_path, "r+") as f:
        jsondata = f.read()
        candidate_high_frequence_dict = json.loads(jsondata)
        # print(candidate_high_frequence_dict)
    print("the len of candidate_high_frequence_dict is: {}".format(len(candidate_high_frequence_dict)))
    data = list()
    id_all_list = list()
    for key,value in candidate_high_frequence_dict.items():
        # print(key, value)
        id_list = key[1:-1].split(', ')
        id_all_list += id_list
        data.append((id_list[0], id_list[1], value[0], value[1]))
    print("the len of the data is : ", len(data))
    writer1.writerows(data)
    csvfile1.close()

    save_dict = list()
    for i in set(id_all_list):
        # print(i, dictionary[int(i)])
        # save_dict.append((i, dictionary[int(i)]))
        save_dict.append((dictionary[int(i)], i))
    print("the len of the dict is : ", len(save_dict))
    writer2.writerows(save_dict)
    csvfile2.close()


