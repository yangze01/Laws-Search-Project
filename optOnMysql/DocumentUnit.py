#-*- coding: UTF-8 -*-
class DocumentsUnit(object):
    '''
        the class in define a dict of userinfo,include:

    '''

    def __init__(self):
        self.documents_unit={
            "_id":"",
            "title":"",
            "court":"",
            "url":"",
            "content":"",
            "criminal":"",
            "date":""
        }#
if __name__ == "__main__":
    document_unit = dict()
    # document_unit["_id"] = 2
    document_unit["title"] = "某某的死刑复核"
    document_unit["court"] = "最高人民法院"
    document_unit["url"] = "http://wenshu.court.gov.cn/CreateContentJS/CreateContentJS.aspx?DocID=eff7f53c-b647-11e3-84e9-5cf3fc0c2c18"
    # print(len(document_unit["url"]))
    document_unit["content"] = "这是一个测试|这是一个测试|这是一个测试|这是一个测试"
    document_unit["criminal"] = "交通肇事罪"
    document_unit["date"] = "2016-12-02"