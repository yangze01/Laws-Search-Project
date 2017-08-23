#coding=utf-8
import pymysql
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class OptOnMysql(object):

    def __init__(self):
        self.db_host = "localhost"
        self.db_user = "root"
        self.db_passowrd = "123"
        # self.db_name = "add_judgment"
        self.db_name = "newdb"

        try:
            self.conn = pymysql.connect(host=self.db_host,user=self.db_user,passwd=self.db_passowrd,db=self.db_name,charset='utf8')
            self.cur = self.conn.cursor()
            print("connect %s %s success"%(self.conn,self.cur))
        except:
            print("connect %s %s fail"%(self.conn, self.cur))
        #-----------------********************-----------------#
    def exeUpdate(self,sql):
        sta = self.cur.execute(sql)
        self.conn.commit()
        return (sta)

    def exeDeleteById(self,sql,ID):
        sta = 0
        sta = self.cur.execute(sql%(int(ID)))
        return(sta)

    def exeQuery(self,sql):
        self.cur.execute(sql)
        self.conn.commit()
        return(self.cur)

    def connClose(self):
        self.cur.close()
        self.conn.close()

if __name__ == "__main__":

    print("test optOnMysql")
    opt_connect = OptOnMysql()
    itit = opt_connect.exeQuery("select * from document;")
    it = itit.fetchall()
    id = 1
    for i in it:
        print(i[0])
        opt_connect.exeQuery("update document set _id = '{0}' where _id = '{1}'".format(id,i[0]))
        id += 1
    # test = opt_connect.exeUpdate("update document set keywords = '{0}' where _id = '{1}'".format("测试,插入,数据",1))
    # test = opt_connect.exeQuery("insert into document ({0}) values ({1})".format('url', "1"))
    opt_connect.connClose()
