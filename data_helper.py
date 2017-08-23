# coding=utf8
from Segment.MySegment import *
from get_element import *
from optOnMysql.DocumentsOnMysql import *
import json
import sys
reload(sys)

sys.setdefaultencoding('utf8')
BasePath = sys.path[0]
def fetch_all_content_result(document_list):
    print("in fetch")
    content_all_list = []
    result_all_list = []
    i = 0
    myseg = MySegment()
    for document in document_list:
        content,result = content_result(myseg, document)
        content_all_list.append(content)
        result_all_list.append(result)
        i += 1
    print(i)
    myseg.close()
    return content_all_list, result_all_list

def get_criminal_list_data(opt_Documents, criminal_list):
    document_id_list = list()
    document_list = list()
    for crim in criminal_list:
        crim_id_list, crim_document_list = get_criminal_data(opt_Documents, crim)
        document_id_list += crim_id_list
        document_list += crim_document_list
    return document_id_list, document_list

def get_criminal_data(opt_Documents, crim):
    '''
    :param crim: 罪名
    :return: document_id_list, document_list
    '''

    document_id_list = list()
    document_list = list()
    print("in get_criminal %s" % crim)
    iter = opt_Documents.findbycriminal(crim)
    print("in segment")
    for it in iter:
        document_id_list.append(it[0])
        # print(it[25])
        document_list.append(it[25])

    return document_id_list, document_list




def read_more_document(filepath_list):

    i = 1
    document = list()
    y = list()
    for filepath in filepath_list:

        current_document_list = read_document(filepath)
        content_all_list, _ = fetch_all_content_result(current_document_list)

        document += content_all_list
        current_y = [i]*len(content_all_list)
        y += current_y
        i += 1
        print(len(document),len(y))
        # print(y)
    return document, y

def read_more_raw_document_list(filepath_list):
    document_list = list()
    for filepath in filepath_list:
        document = read_document(filepath)
        document_list += document
    return document_list

def read_document(file_path):
    return_document = list()
    # i = 0

    for line in open(file_path):
        # print(i)
        # i += 1
        # print(line)
        line = json.loads(line.decode('utf8'))
        print(line)
        decode_line = ' '.join(line['content'])
        # print(decode_line)
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
    # myseg.close()
    return content_wordlist, result_wordlist

def seg_document(document_list):
    content_list = []
    result_list = []

    myseg = MySegment()
    i = 0
    for document in document_list:
        print(i)
        content_wordlist, result_wordlist = content_result(myseg, document)
        content_list.append(content_wordlist)
        result_list.append(result_wordlist)
        i += 1
    return content_list,result_list

def save_seg_document(content_list,result_list, i):
    seg_dict = dict()
    filepath = BasePath + "/data/judgment" + str(i) + ".txt"
    seg_dict["content"] = content_list
    seg_dict["result"] = result_list
    encode_json = json.dumps(seg_dict, ensure_ascii = False)
    with open(filepath, "w+") as f:
        f.write(encode_json)
    print("seg document saved as {}".format(filepath))

def read_one_seg_document(file_path):
    with open(file_path,"r+") as f:
        jsondata = f.read()
        jsondict = json.loads(jsondata)
    content_list = jsondict['content']
    return content_list

def read_seg_document_list(file_path_list):
    all_document_list = list()
    label = list()
    i = 1
    for filepath in file_path_list:
        document_list = read_one_seg_document(filepath)
        all_document_list += document_list
        y = [i]*len(document_list)
        label += y
        i = i + 1
        print("the read document size")
        print(len(document_list),len(y))
    return all_document_list, label





if __name__ == "__main__":

    a = [BasePath + "/data/judgment" + str(i) + ".txt" for i in range(1,8)]




    # filepath = BasePath + "/data/judgment_trafficking.txt"
    # document_list = read_document(filepath)
    # content_list, result_list = seg_document(document_list)
    # save_seg_document(content_list, result_list, 7)
    # content_list = read_one_seg_document(BasePath + "/data/judgment7.txt")
    # print(len(content_list))
    # for i in content_list:
        # print(' '.join(i))



