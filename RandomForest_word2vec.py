#coding=utf8
from optOnMysql.DocumentsOnMysql import *
from sklearn.externals import joblib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from Segment.MySegment import *
from Segment.MyPosTag import *
import requests
from data_helper import *
from My_BasePath import *
import matplotlib as mpl
import numpy as np
from pylab import *
import gensim
import heapq

myfont = mpl.font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
mpl.rcParams['axes.unicode_minus'] = False
np.seterr(divide='ignore', invalid='ignore')
sys.setdefaultencoding('utf8')
"""
    加载相关数据
"""
fv_Word2Vec = BasePath + "/word2vec_model/fv_Word2Vec_add_finance"#_test_min_count5"
w2v_model = gensim.models.Word2Vec.load(fv_Word2Vec)
w2v_model_min_count5 = gensim.models.Word2Vec.load(fv_Word2Vec + "_min_count_5")
# # id索引
document_all_id_list = np.loadtxt(BasePath + "/data/document_full_finance_index.txt")# 40000 + 8000条数据版本
print("load document index finished, the length is : {}".format(len(document_all_id_list)))
# 语料向量
x_sample = np.loadtxt(BasePath + "/word2vec_model/corpus_w2v_full_finance_average.txt")
print("load the corpus vector in : {}".format(BasePath + "/word2vec_model/corpus_w2v_full_average.txt"))
# # 随机森林训练
clf_filepath = BasePath + "/data/clf_model_full_average.m"
if os.path.exists(clf_filepath):
    print("the model already exists in :{}".format(clf_filepath))
    clf = joblib.load(clf_filepath)
else:
    print("No model loaded!")

# 分词模块
myseg = MySegment()
opt_Document = DocumentsOnMysql()

def plot_confusion_matrix(cm, title = "Confusion matrix", cmap = plt.cm.Blues):
    classes = [       u'交通肇事罪',
                      u'过失致人死亡罪',
                      u'故意杀人罪',
                      u'故意伤害罪',
                      u'过失致人重伤罪',
                      u'抢劫罪',
                      u'诈骗罪',
                      u'拐卖妇女儿童罪']
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontproperties=myfont)
    plt.yticks(tick_marks, classes,fontproperties=myfont)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

def rf_similarity(path_vec):
    fea_size = len(path_vec)
    sim_vec = np.zeros([fea_size, fea_size])
    k = 0
    for i in range(0, fea_size):
        for j in range(i, fea_size):
            print(k)
            k = k+1
            num_path = np.sum(path_vec[i] & path_vec[j])
            sim_vec[i][j] = num_path
            sim_vec[j][i] = num_path
    return sim_vec/sim_vec.diagonal().T

def sentence2vec(model, sentence, randomvec = None, vec_type = "average"):
    # print(vec_type)
    if randomvec == None:
        randomvec = np.random.normal(size = 100)
    len_word = len(set(sentence))
    tmp_num = np.zeros(100)
    if vec_type == "average":
        if (len_word == 0):
            return np.zeros(100)
        for word in set(sentence):
            try:
                tmp_num += model[word.decode('utf8')]
            except:
                tmp_num += randomvec
        tmp_num = tmp_num / len_word

    elif vec_type == "minmax":
        # print("in minmax")
        if (len_word == 0):
            return np.zeros(200)
        for word in set(sentence):
            try:
                tmp_num = np.vstack((tmp_num, model[word.decode('utf8')]))
            except:
                tmp_num = np.vstack((tmp_num, randomvec))
        tmp_num = np.hstack((np.min(tmp_num, axis = 0), np.max(tmp_num, axis = 0)))

    return tmp_num
def sentences2docvec(model, sentences, vec_type = "average"):
    i = 0
    random_vector = np.random.normal(size = 100)
    corpus_vec = list()
    for sentence in sentences:
        # if (i == 2658):
        #     print(1)
        tmp_num = sentence2vec(model, sentence, randomvec = random_vector, vec_type = vec_type)
        # len_word = len(set(sentence))
        print(i)

        # tmp_num = np.zeros(300)
        # for word in set(sentence):
        #     try:
        #         tmp_num += model[word.decode('utf8')]
        #     except:
        #         tmp_num += random_vector
        # tmp_num = tmp_num/len_word
        corpus_vec.append(tmp_num)
        i = i + 1
    np.savetxt(BasePath + "/word2vec_model/corpus_w2v_full_finance_" + vec_type + ".txt", np.array(corpus_vec))

def corpus2word2vec(x_data):
    # w2v_model = load_model()
    sentences2docvec(w2v_model, x_data)

def get_candidate(topn, query_vec, corpus_vec):
    vec_sim = np.dot(query_vec, corpus_vec.T)  / (np.linalg.norm(corpus_vec, axis = 1) * np.linalg.norm(query_vec))
    topn_candidate = heapq.nlargest(topn, range(len(vec_sim)), vec_sim.take)
    return topn_candidate, vec_sim[topn_candidate]
def get_full_text_candidate(topn, seg_sentence):
    request_data = requests.post("http://10.168.103.101:8000/document_solr/document/list.controller",data={'queryString':seg_sentence})
    decode_json = json.loads(request_data.content)
    topn_candidate = decode_json['ids']
    full_text_sim = decode_json['scores']
    if len(topn_candidate) >topn:
        return np.array(topn_candidate)[0:topn], np.array(full_text_sim)[0:topn]
    else:
        return np.array(topn_candidate), np.array(full_text_sim)

def get_clf_sim(clf_model, seg_sentence_vec, candidate_vec, topn_candidate_index):
    path_of_sample, _ = clf.decision_path(candidate_vec[topn_candidate_index])
    path_of_seg_sentence, _ = clf.decision_path(seg_sentence_vec)
    seg_sentence_array = path_of_seg_sentence.toarray()
    sample_array = path_of_sample.toarray()
    clf_sim = np.sum(seg_sentence_array & sample_array, axis=1) / float(np.sum(seg_sentence_array & seg_sentence_array))
    if len(clf_sim >50):
        top_clf_index = heapq.nlargest(50, range(len(clf_sim)), clf_sim.take)
    else:
        top_clf_index = heapq.nlargest(len(topn_candidate_index), range(len(clf_sim)), clf_sim.take)
    # return np.array(topn_candidate_index)[top_clf_index]
    return top_clf_index, clf_sim

def tsne_trans(input_vector):
    X_tsne = TSNE(learning_rate = 100).fit_transform(input_vector)
    return X_tsne

def get_sim_sentence(clf_model, seg_sentence, x_sample):
    seg_sentence_vec = sentence2vec(w2v_model, seg_sentence, vec_type = "average")
    """
        top_candidate_index返回vec_sim中对应的n个large
        top_clf_index是候选中的索引
    """

    # 全文检索候选
    candidate_index, candidate_vec_score = get_full_text_candidate(300,seg_sentence)
    if(len(candidate_index) != 0):

        top_clf_index, clf_sim = get_clf_sim(clf_model, seg_sentence_vec, x_sample, candidate_index - 1)
        print("~~~~~~~~~~~~")
        # print(candidate_index)
        result_vec = x_sample[candidate_index-1]
        if (len(candidate_index) == 1):
            result_vec = np.row_stack((result_vec, x_sample[0]))

        tsne_vec = tsne_trans(result_vec)

        tsne_vec_dict = {top_clf_index[i]:tsne_vec[i] for i in range(0, len(top_clf_index))}

        document_ret_dict = [{'id' : document_all_id_list[candidate_index[i] - 1],
                              'final_sim' : clf_sim[i],
                              'vec_sim' : candidate_vec_score[i],
                              'position' : {
                                  'x' : tsne_vec_dict[i][0],
                                  'y' : tsne_vec_dict[i][1]}
                             } for i in top_clf_index]
        document_ret_dict.sort(key=lambda x: x["final_sim"],reverse = True)
        return document_ret_dict

    else:
        topn_candidate_index, candidate_vec_sim = get_candidate(300, seg_sentence_vec, x_sample)
        top_clf_index, clf_sim = get_clf_sim(clf_model, seg_sentence_vec, x_sample, topn_candidate_index)
        result_vec = x_sample[np.array(topn_candidate_index)[top_clf_index]]
        tsne_vec = tsne_trans(result_vec)
        tsne_vec_dict = {top_clf_index[i]: tsne_vec[i] for i in range(0, len(top_clf_index))}
        document_ret_dict = [{'id': document_all_id_list[topn_candidate_index[i]],
                              'final_sim': clf_sim[i],
                              'vec_sim': candidate_vec_sim[i],
                              'position': {
                                  'x': tsne_vec_dict[i][0],
                                  'y': tsne_vec_dict[i][1]}

                              } for i in top_clf_index]
        document_ret_dict.sort(key=lambda x: x["final_sim"],reverse = True)
        # print(document_ret_dict)
        return document_ret_dict

def get_w2v_key(word):
    # print(word)
    word_tuple_list = w2v_model_min_count5.most_similar(word.decode('utf8'), topn=5)
    ret_dict = {'key': [wordtuple[0] for wordtuple in word_tuple_list],
                'value': [wordtuple[1] for wordtuple in word_tuple_list]}
    # print(ret_dict)
    return ret_dict

def get_keywords(seg_sentence):
    print(1)
    return_word_list = list()
    return_relation_list = dict()
    for word in set(seg_sentence):
        try:
            # word_tuple_list = w2v_model_min_count5.most_similar(word.decode('utf8'), topn=5)
            relation_dict = get_w2v_key(word)
            word_dict_list = [{'word' : key,
                               'cluster' : word.encode('utf8')} for key in relation_dict['key']]

            return_word_list += word_dict_list
            tmp_list = list()
            relation_word_dict = dict(zip(relation_dict['key'], relation_dict['value']))
            for key,value in relation_word_dict.items():
                print(key)
                print(value)

            for tuple in zip(relation_dict['key'], relation_dict['value']):
                tmp_list.append({'key':tuple[0], 'value':tuple[1]})
            return_relation_list[word] = tmp_list
            # return_relation_list.append({word:relation_dict})
            # return_relation_list.append({relation_dict})

        except:
            continue

    return return_word_list, return_relation_list

def impl_sim(search_type, sentence):
    seg_sentence = myseg.sen2word(sentence.encode('utf8'))

    return_word_list, return_relation_list = get_keywords(seg_sentence)
    document_ret_dict = get_sim_sentence(clf, seg_sentence, x_sample)
    print(document_ret_dict)
    return_json_list = {'keywords':return_word_list,
                        'result':document_ret_dict,
                        'relation':return_relation_list}

    return return_json_list

def resorted_data(document_index, document_vec):
    merge_data = zip(document_index, document_vec)
    sorted_merge = sorted(merge_data, key = lambda t: t[0])
    document_all_id_list2, x_sample2 = zip(*sorted_merge)
    return document_all_id_list2, x_sample2
if __name__ == "__main__":
    while(True):
        print("请输入一句话或空格间隔的关键词，回车结束： ")
        sentence = raw_input()
        document_ret_dict = impl_sim(2, sentence)
        j = 1
        for json_obj in document_ret_dict['result']:
            # print(json_obj['id'])
            # print(json_obj)
            print("----------------------- 第" + str(j) + "名匹配文档： -----------------------")
            print("----------------------- 第" + str(j) + "名匹配文档的clf相似度: {}------------------------------------".format(json_obj['final_sim']))
            print("----------------------- 第" + str(j) + "名匹配文档的vec相似度: {}------------------------------------".format(json_obj['vec_sim']))

            print("document id {}".format(json_obj['id']))
            # print('\n'.join(document_list[document_tuple[0]].split('|')))
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print('\n'.join(opt_Document.getById(json_obj['id'])[25].split('|')))
            j += 1



















