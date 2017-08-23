#coding=utf8
import sys
BasePath = sys.path[0]
from Segment.MyPosTag import *
from Segment.MySegment import *
from data_helper import *

def get_body(document):
    pattern_person = re.compile(u"(被告人|罪犯).*?\。")
    search_result = re.search(pattern_person, document)
    if search_result:
        return_str = search_result.group()
    else:
        return_str = ""
    return return_str

def get_behavior(document):
    # pattern_behavior = re.compile(u"(本院认为).*?\。")
    pattern_behavior = re.compile(u"(本院认为).*(依照)")
    search_result = re.search(pattern_behavior, document) #
    if search_result:
        return_str = search_result.group()
    else:
        return_str = ""
    return return_str

def get_result(document):
    pattern_result = re.compile(u"(判决如下|裁定如下|判处如下|决定如下|判决意见如下).*?(如不服)")
    search_result = re.search(pattern_result, document)
    if search_result:
        return_str = search_result.group()
    else:
        pattern_result = re.compile(u"(判决如下|裁定如下|判处如下|决定如下|判决意见如下).*?\。")
        search_result = re.search(pattern_result, document)
        if search_result:
            return_str = search_result.group()
        else:
            return_str = ""
    return return_str #

def get_details(document):
    pattern_details = re.compile(u".*?(上述事实)")
    search_result = re.search(pattern_details, document)
    if search_result:
        # print("------------------------------------")
        return_str = search_result.group()
    else:
        return_str = document[0:3*200]
    return return_str

def get_legal_behaviors_words(myseg, document):
    behavior_not = "不予"
    # behavior1_set = {"自首", "坦白", "累犯"}
    # behavior2_set = {"谅解", "和解"}
    behavior_set = {"自首", "坦白", "累犯", "谅解", "和解"}
    return_word_set = set()
    behavior_senlist = myseg.paraph2sen(get_behavior(document))
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for sentence in behavior_senlist:
        # print("+++++++++++++++++++++++++++++++++++++++++++++++")
        # print(sentence)
        sen_word_set = set(myseg.sen2word(sentence))
        # print(' '.join(list(sen_word_set)))
        if behavior_not not in sen_word_set:
            # print(1)
            word_left = sen_word_set & behavior_set
            # print(word_left)
            return_word_set |= word_left
            # print(return_word_set)
            # print(' '.join(list(word_left)))
    # print(' '.join(list(return_word_set)))
    return list(return_word_set)

def get_details_words(myseg, mypos, document):
    behavior_set = {"自首", "坦白", "累犯", "谅解", "和解"}
    # print("+++++++++++++++++++++++++++++++")

    decode_document = get_details(document)
    # print(decode_document)
    details_words = myseg.sen2word(decode_document.encode('utf8'))
    # print(' '.join(details_words))
    sdetails_words = list(set(mypos.words2pos(details_words, ['n', 'nl', 'ns', 'v'])) - behavior_set)
    # print(' '.join(sdetails_words))
    return sdetails_words

if __name__ == "__main__":
    file_path = BasePath + "/data/judgment_yishen.txt"
    print(file_path)
    myseg = MySegment()
    mypos = MyPostagger()
    document_list = read_document(file_path)
    for document in document_list:
        print("___________________________________________________")
        print(get_details(document))
    myseg.close()
    mypos.close()




