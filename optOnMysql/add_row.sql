alter table document ADD
keywords varchar(10000),
case_type varchar(200),
case_num varchar(200),
public_prosecution varchar(200);


/*
  create table of index
 */
create table sen_index( _id int unsigned not null primary key auto_increment, word_list LongText) ENGINE=InnoDB DEFAULT CHARSET=utf8;
create table word_index( _id int unsigned not null primary key auto_increment, word varchar(100)) ENGINE=InnoDB DEFAULT CHARSET=utf8;
create table sen_doc( _id int unsigned not null primary key auto_increment, sen_id int unsigned, dic_id int unsigned) ENGINE=InnoDB DEFAULT CHARSET=utf8;


/*
  mysql dump
 */
mysqldump -u root -p --routines --default-character-set=utf8 --databases judgment > judgment_with_abstract.sql
