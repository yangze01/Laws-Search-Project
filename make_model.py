#coding=utf8
from My_BasePath import *
import gensim
import datetime
import numpy as np
from scipy import io
import matplotlib as mpl
from sklearn.externals import joblib
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from scipy.sparse.csr import csr_matrix
from optOnMysql.DocumentsOnMysql import *
from data_helper import *
import matplotlib.pyplot as plt
import pickle
from seg_main import *
myfont = mpl.font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
mpl.rcParams['axes.unicode_minus'] = False
np.seterr(divide='ignore', invalid='ignore')


def dev_sample(x_sample, y_sample, dev_sample_percentage):
    np.random.seed(10)
    print(len(y_sample))
    shuffle_indices = np.random.permutation(np.arange(len(y_sample)))
    print(shuffle_indices)
    x_shuffled = x_sample[shuffle_indices]
    y_shuffled = y_sample[shuffle_indices]

    dev_sample_index = -1 * int(dev_sample_percentage * float(len(x_sample)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    return x_train, x_test, y_train, y_test

def plot_confusion_matrix(cm, title = "Confusion matrix", cmap = plt.cm.Blues):
    classes = [       u'交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                      u'过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      u'故意杀人罪',
                      u'故意伤害罪',
                      u'过失致人重伤罪',
                      u'抢劫罪',
                      u'诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      u'拐卖妇女儿童罪']
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45, fontproperties = myfont)
    plt.yticks(tick_marks, classes, fontproperties = myfont)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted label")

def sentence2vec(model, sentence, randomvec = None, vec_type = "average"):
    if randomvec == None:
        randomvec = np.random.normal(size = 100)
    len_word = len(set(sentence))
    tmp_num = np.zeros(100)
    if vec_type == "average":
        if(len_word == 0):
            return np.zeros(100)
        for word in set(sentence):
            try:
                tmp_num += model[word.decode('utf8')]
            except:
                tmp_num += randomvec
        tmp_num = tmp_num / len_word
    elif vec_type == "minmax":
        if(len_word == 0):
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
        tmp_num = sentence2vec(model, sentence, randomvec = random_vector, vec_type = vec_type)
        print(i)
        corpus_vec.append(tmp_num)
        i = i + 1
    np.savetxt(BasePath + "/word2vec_model/corpus_w2v_" + vec_type + ".txt", np.array(corpus_vec))


def content_resultforword2vec(myseg, document):
    judge_pattern = re.compile(u"(.*)((判决如下|裁定如下|判处如下|判决)(.*))")
    matcher1 = re.match(judge_pattern, document)#在源文本中搜索符合正则表达式的部分
    if matcher1:
        content_wordlist = myseg.paraph2word(get_details(matcher1.group(1)))
        result_wordlist = myseg.paraph2word(matcher1.group(2))
    else:
        content_wordlist = myseg.paraph2word(get_details(document))
        result_wordlist = []
    return content_wordlist, result_wordlist
class TwoDict(object):
    def __init__(self):
        self.word2id = dict()
        self.id2word = dict()



def get_criminal_dict(opt_document):
    it = opt_document.getCriminalOnSql()
    word2id = dict()
    count = 0
    for crim in it:
        # print(crim[0])
        word2id[crim[0]] = count
        count+=1
    return word2id

def seg_data(opt_document, myseg, mypos):

    word2id = get_criminal_dict(opt_document)
    print(word2id)

    content_list = list()
    result_list = list()
    id_list = list()
    criminal_list = list()
    iter = opt_document.findall()
    index = 0
    save_index = 0
    for it in iter:
        print(index)
        content_wordlist, result_wordlist = content_resultforword2vec(myseg, it[25])
        content_wordlist = mypos.words2pos(content_wordlist, ['n', 'nl', 'ns', 'v'])
        result_wordlist = mypos.words2pos(result_wordlist, ['n', 'nl', 'ns', 'v'])

        content_list.append(content_wordlist)
        result_list.append(result_wordlist)
        id_list.append(it[0])
        criminal_list.append(word2id[it[26]])

        if index % 10000 == 0 and index != 0:
            print("~~~~~~~~~~~~~~~{}~~~~~~~~~~~~~~~~~".format(index))
            save_dict = {'id_list': id_list,
                         'criminal_list': criminal_list,
                         'content_wordlist': content_list,
                         'result_wordlist': result_list}
            with open(BasePath + "/seg_corpus/data_corpus"+str(save_index)+".json", 'wb') as fp:
                json.dump(save_dict, fp, ensure_ascii=False)
                print("save data in the file {}".format(BasePath + "/seg_corpus/data_corpus"+str(save_index)+".json"))
            save_index += 1

            criminal_list = list()
            content_list = list()
            result_list = list()
            id_list = list()
        index += 1

    if len(content_list) > 0:
        print("the len of the data left is: {}".format(len(content_list)))
        save_dict = {'id_list': id_list,
                     'criminal_list': criminal_list,
                     'content_wordlist': content_list,
                     'result_wordlist': result_list}
        with open(BasePath + "/seg_corpus/data_corpus" + str(save_index) + ".json", 'wb') as fp:
            json.dump(save_dict, fp, ensure_ascii=False)
            print("save data in the file {}".format(BasePath + "/seg_corpus/data_corpus" + str(save_index) + ".json"))

    print("finish all the data segment, the number of files is: {}".format(index))

def read_from_data_corpus(filepath):
    with open(filepath, 'rb') as fp:
        decode_json = json.load(fp)
    return decode_json

def train_w2v(filepath_list, model_savepath):
    for index, filepath in enumerate(filepath_list):
        print(filepath)
        x_data = read_from_data_corpus(filepath)['content_wordlist']
        if index == 0:
            model = gensim.models.Word2Vec(x_data, size=100, window=5, min_count=5, workers=20)
        else:
            model.train(x_data)
    model.save(model_savepath)
    return model

def get_filepath_list_from_dir(dirpath):
    filepath_list = []
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            filepath_list.append(os.path.join(root, file))
    return filepath_list
def corpus2vec(w2v_model, filepath_list, vec_type = "average"):
    print("~~~~~~~~~~~")
    random_vector = np.random.normal(size = 100)
    for filepath in filepath_list:
        data = read_from_data_corpus(filepath)
        # file_name = filepath.split('/')[-1]
        filepath, file_name = '/'.join(filepath.split('/')[0:-2]), filepath.split('/')[-1].split('.')[0]
        id_list, x_data, y_data = data['id_list'],\
                                  data['content_wordlist'],\
                                  data['criminal_list']
        i = 0
        corpus_vec = list()
        for sentence in x_data:
            print(i)
            tmp_num = sentence2vec(w2v_model, sentence, randomvec = random_vector, vec_type = vec_type)
            corpus_vec.append(tmp_num)
            i += 1
        np.savetxt(filepath + "/w2v_corpus/w2v_" + file_name + ".txt", np.array(corpus_vec))
        save_dict = { 'id_list': data['id_list'],
                      'criminal_list': data['criminal_list']
                    }
        with open(filepath + "/w2v_corpus/index_"+ file_name + ".json", 'wb') as fp:
            json.dump(save_dict, fp, ensure_ascii=False)
        print("save data in the file {}".format(filepath + "/w2v_corpus/index_"+ file_name + ".json"))

def load_model(model_filepath):
    model = gensim.models.Word2Vec.load(model_filepath)
    return model

# def randomforest_partition_train(x_data, y_data, save_model_path):

def randomforest_train(data_path, save_model_path):
    num_topics = 100
    dev_sample_percentage = .2
    index_list = list()
    w2v_list = list()
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                w2v_list.append(file)
            elif os.path.splitext(file)[1] == '.json':
                index_list.append(file)
    index_list.sort()
    w2v_list.sort()
    x_sample = list()
    y_sample = list()
    for index_path, w2v_path in zip(index_list, w2v_list):
        with open(BasePath + "/w2v_corpus/" + index_path, 'rb') as fp:
            y_sample += json.load(fp)['criminal_list']

        x_sample += list(np.loadtxt(BasePath + "/w2v_corpus/" + w2v_path))

    x_sample = np.array(x_sample)
    y_sample = np.array(y_sample)
    print("the shape of x_sample and y_sample")
    print(x_sample.shape)
    print(y_sample.shape)
    x_train, x_test, y_train, y_test = dev_sample(x_sample, y_sample, dev_sample_percentage)
    if os.path.exists(save_model_path):
        print("the model already exists.")
    else:
        print("the model doesn't exists.")
        clf = RandomForestClassifier(n_estimators=100, bootstrap = True, oob_score = False , n_jobs = 16)
        clf.fit(x_train, y_train)
        joblib.dump(clf, save_model_path)
    clf_pre = clf.predict(x_test)
    acc = (clf_pre == y_test).mean()
    print("精度为：")
    print(acc)
    return clf

def get_one_rfpath(model_path, vec):
    if os.path.exists(model_path):
        print("the model already exists.")
        clf = joblib.load(model_path)
    else:
        print("the model doesn't exists.")
        return None

    path = clf.decision_path(vec)
    return path


def save_randomforest_path(model_path, w2v_path):
    if os.path.exists(model_path):
        print("the model already exists.")
        clf = joblib.load(model_path)
    else:
        print("the model doesn't exists.")
        return None

    w2v_list = list()
    for root, dirs, files in os.walk(w2v_path):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                w2v_list.append(file)
    w2v_list.sort()
    for w2v_name in w2v_list:
        filename = BasePath + "/w2v_corpus/" + w2v_name
        w2v_vec = np.loadtxt(filename)
        print(w2v_vec.shape)
        path_of_sample, _ = clf.decision_path(w2v_vec)
        save_file_path = BasePath + "/rf_path/" + "path_" + w2v_name.split('.')[0] + ".mtx"
        io.mmwrite(save_file_path, path_of_sample)



if __name__ == "__main__":
    # print(1)
    shuffle_indices = np.random.permutation(5)
    print(shuffle_indices)

    # 将数据库中的数据进行分词，存储为文件
    # opt_document = DocumentsOnMysql()
    # myseg = MySegment()
    # mypos = MyPostagger()
    # seg_data(opt_document, myseg, mypos)

    # 训练词向量
    # filepath_list = get_filepath_list_from_dir(BasePath + "/seg_corpus")
    # filepath_list.sort()
    # print(filepath_list)
    # model_savepath = BasePath + "/model/corpus_w2v_model"
    # model = train_w2v(filepath_list, model_savepath)

    # 将语料转换为向量，存储到本地
    # model_filepath = BasePath + "/model/corpus_w2v_model"
    # model = load_model(model_filepath)
    # filepath_list = get_filepath_list_from_dir(BasePath + "/seg_corpus")
    # filepath_list.sort()
    # corpus2vec(model, filepath_list)

    # 训练随机森林
    # data_dir = BasePath + "/w2v_corpus"
    # save_model_name = BasePath + "/model/rf_model"
    # clf_model = randomforest_train(data_dir, save_model_name)

    # 获取随机森林路径
    # rf_model_path = BasePath + "/model/rf_model"
    # w2v_corpus = np.loadtxt(BasePath + "/w2v_corpus/w2v_data_corpus0.txt")
    # w2v_path = BasePath + "/w2v_corpus"
    # vec = w2v_corpus[0]
    # path = get_one_rfpath(rf_model_path, vec)
    # save_randomforest_path(rf_model_path, w2v_path)









