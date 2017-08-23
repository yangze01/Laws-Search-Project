# coding=utf8
from Segment.MySegment import *
from Segment.MyPosTag import *
from get_element import *
from data_helper import *
import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')
BasePath = sys.path[0]

def read_document(file_path):
    return_document = list()
    # i = 0
    for line in open(file_path):
        # print(i)
        # i += 1
        # print(line)
        line = json.loads(line.decode('utf8'))
        decode_line = ''.join(line['content'])
        return_document.append(decode_line.decode('utf8'))
    return return_document

def content_result(myseg, document):
    judge_pattern = re.compile(u"(.*)((判决如下|裁定如下|判处如下|判决)(.*))")
    matcher1 = re.match(judge_pattern, document)#在源文本中搜索符合正则表达式的部分
    if matcher1:
        content_wordlist = myseg.paraph2word(matcher1.group(1))
        result_wordlist = myseg.paraph2word(matcher1.group(2))
    else:
        content_wordlist = myseg.paraph2word(document)
        result_wordlist = []

    return content_wordlist, result_wordlist

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


def seg_document(document_list):
    content_list = []
    result_list = []
    i = 0
    myseg = MySegment()
    mypos = MyPostagger()
    for document in document_list:
        # print(i)
        # i = i + 1
        content_wordlist, result_wordlist = content_result(myseg, document)

        content_wordlist = mypos.words2pos(content_wordlist, ['n', 'nl', 'ns', 'v'])
        result_wordlist = mypos.words2pos(result_wordlist, ['n', 'nl', 'ns', 'v'])
        content_list.append(content_wordlist)
        result_list.append(result_wordlist)
    myseg.close()
    mypos.close()
    print("----------------------------------------------")
    return content_list,result_list

def seg_documentforword2vec(document_list):
    content_list = []
    result_list = []
    i = 0
    myseg = MySegment()
    mypos = MyPostagger()
    for document in document_list:
        # print(i)
        # i = i + 1
        content_wordlist, result_wordlist = content_resultforword2vec(myseg, document)

        content_wordlist = mypos.words2pos(content_wordlist, ['n', 'nl', 'ns', 'v'])
        result_wordlist = mypos.words2pos(result_wordlist, ['n', 'nl', 'ns', 'v'])
        content_list.append(content_wordlist)
        result_list.append(result_wordlist)
    myseg.close()
    mypos.close()
    print("----------------------------------------------")
    return content_list, result_list


def save_seg_document(content_list,result_list, i):
    seg_dict = dict()
    # filepath = BasePath + "/data/judgment" + str(i) + "wordforword2vec" ".txt"
    filepath = BasePath + "/data/judgment" + str(i) + ".txt"
    print(filepath)
    seg_dict["content"] = content_list
    seg_dict["result"] = result_list
    encode_json = json.dumps(seg_dict, ensure_ascii = False)
    with open(filepath, "w+") as f:
        f.write(encode_json)
    print("seg document saved as {}".format(filepath))

def read_seg_document(file_path):
    with open(file_path,"r+") as f:
        jsondata = f.read()
        jsondict = json.loads(jsondata)
    content_list = jsondict['content']
    return content_list

if __name__ == "__main__":

    # filepath_list = [BasePath + "/data/judgment.txt", \
    #                  BasePath + "/data/judgment_Death_by_obsolescence.txt", \
    #                  BasePath + "/data/judgment_kill.txt", \
    #                  BasePath + "/data/judgment_negligence_caused_serious_injury.txt", \
    #                  BasePath + "/data/judgment_robbery.txt", \
    #                  BasePath + "/data/judgment_scam.txt", \
    #                  BasePath + "/data/judgment_trafficking.txt"]
    criminal_list = ['交通肇事罪',  # 危险驾驶罪（危险 驾驶罪）
                     '过失致人死亡罪', # 故意杀人罪（故意 杀人 杀人罪） 故意伤害罪（故意 伤害 伤害罪）
                      '故意杀人罪',
                      '故意伤害罪',
                      '过失致人重伤罪',
                      '抢劫罪',
                      '诈骗罪', #（诈骗 诈骗罪 诈骗案）
                      '拐卖妇女儿童罪'
                      ]

    # criminal_list = [u"交通肇事罪",
    #                  # u"过失致人死亡罪",
    #                  # u"故意杀人罪",
    #                  # u"故意伤害罪",
    #                  # u"抢劫罪",
    #                  u"诈骗罪"]#,
    #                  # u"拐卖妇女儿童罪"]

    # myseg = MySegment()
    # word_list = myseg.sen2word('张某酒后驾车逃逸'.encode('utf8'))
    # print(' '.join(word_list))
    # print(word_list)


    # i = 0
    # opt = DocumentsOnMysql() #
    # doucment_id_list, document_list = get_criminal_data(opt, criminal_list[i])
    # content_list, result_list = seg_documentforword2vec(document_list)
    # print(len(content_list))
    # save_seg_document(content_list, result_list, 'full_finance_'+str(i)+'_')
    # opt.connClose()

    i = 8
    opt = DocumentsOnMysql()
    doucment_id_list, document_list = get_criminal_data(opt, criminal_list[i])
    content_list, result_list = seg_document(document_list)
    print(len(content_list))
    save_seg_document(content_list, result_list, 'full_finance_' + str(i) + '_')
    opt.connClose()

