#coding=utf8
import sys
import time
reload(sys)
sys.setdefaultencoding('utf8')
from optOnMysql.DocumentsOnMysql import *
from optOnMysql.DocumentUnit import *
import json
BasePath = sys.path[0]
def is_valid_date(str):
  '''判断是否是一个有效的日期字符串'''
  try:
    time.strptime(str, "%Y-%m-%d")
    return True
  except:
    return False#

def document_format(line, criminal):
    line = json.loads(line.decode('utf8'))
    document_unit = dict()
    document_unit["title"] = line['title']
    # print(len(document_unit['title']))
    document_unit["court"] = line['court']
    document_unit["url"] = line['url']
    document_unit["content"] = '|'.join(line['content']).encode('utf8')
    # print(len(document_unit["content"]))
    document_unit["criminal"] = criminal
    if(is_valid_date(line["date"])):
        document_unit["date"] = line['date']
    else:
        document_unit["date"] = "0000-00-00"
    return document_unit

def save_document2mysql(file_path, criminal):
    opt = DocumentsOnMysql()
    i = 0
    for line in open(file_path):

        print(i)
        i = i + 1
        document_unit = document_format(line, criminal)
        opt.insertOneDocuments(document_unit)
    opt.connClose()
    print(u"finished")




if __name__ == "__main__":
    opt = DocumentsOnMysql()
    # opt.insertOneDocuments(document_unit)
    # print(opt)
    opt.findById("1")
    a = opt.findall()
    for i in a :
        print(i)
    opt.connClose()
    # file_path =  BasePath + "/../data/judgment_trafficking.txt"
    # save_document2mysql(file_path,u"拐卖妇女儿童罪")
