#coding=utf8
from My_BasePath import *
import numpy as np
import matplotlib as mpl
from data_helper import *
import matplotlib.pyplot as plt
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



if __name__ == "__main__":
    print(1)
    shuffle_indices = np.random.permutation(5)
    print(shuffle_indices)

    num_topics = 100
    dev_sample_percentage = .3
    filepath_list = [BasePath + "/data/judgment" +"full_finance_" +str(i)+ "_" + ".txt" for i in range(0,8)]
    x_data, y_data = read_seg_document_list(filepath_list)

































