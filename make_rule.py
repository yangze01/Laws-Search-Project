#coding=utf8
import tensorflow as tf
import numpy as np
import os
import sys
import re
from Segment.MySegment import *
from optOnMysql.DocumentsOnMysql import *
import jieba
import json
reload(sys)
sys.setdefaultencoding('utf8')
import os
BasePath = sys.path[0] + "/data/"


chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
        u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
        u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
        u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
        u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
        u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
        u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
        u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
        u'５':5, u'６':6, u'７':7, u'８':8, u'９':9}

def chinese2num (chinese_digits, encoding="utf-8"):
    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits.decode (encoding)

    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result

def get_case_reason(data):
    # print(type(data))
    # print(data)
    # pattern_fa_tiao = re.compile("(》|、)(.*?)第([一 二 三 四 五 六 七 八 九 十 百].*?)条")
    # print(data)
    pattern_fa_tiao = re.compile(u"(》|、)(.*?)第([零 一 二 三 四 五 六 七 八 九 十 百].*?)条")
    pattern_fa_tiao2 = re.compile(u"第([0 1 2 3 4 5 6 7 8 9].*?|[零 一 二 三 四 五 六 七 八 九 十 百][零 一 二 三 四 五 六 七 八 九 十 百].*?)条")
    pattern_fa_tiao3 = re.compile(u"第([零 一 二 三 四 五 六 七 八 九 十 百][零 一 二 三 四 五 六 七 八 九 十 百].*?)条")

    search_result = re.findall(pattern_fa_tiao3, data)
    # print(type(search_result))
    # print(' '.join(search_result))
    if search_result:
        # print search_result
        return search_result
    else:
        return -1

def get_result(data):
    pattern_divided = re.compile(u'(依据|依照)《中华人民共和国刑法》(.*?)(《|。)')
    search_result = re.search(pattern_divided, data)
    if search_result:
        return search_result.group()
    else:
        return ""

if __name__ == "__main__":

    opt_doc = DocumentsOnMysql()
    it = opt_doc.findall()
    opt = OptOnMysql()
    for i in it:
        doc_id = i[0]
        content = i[25]
        result = get_result(content)
        case_reason = get_case_reason(result)
        if case_reason == -1:
            continue
        num_reason = [chinese2num(chn) for chn in case_reason if chinese2num(chn) != 0]
        print(num_reason)
        for reason in num_reason:
            opt.exeQuery("insert into doc2rule(doc_id, rule_id) values('{0}', '{1}')".format(doc_id, reason))

    opt.connClose()
    opt_doc.connClose()


































