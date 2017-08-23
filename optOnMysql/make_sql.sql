create table document(
  _id int unsigned not null primary key auto_increment,
  title varchar(500),
  court varchar(200),
  case_num varchar(200),
  type varchar(200),
  case_reason varchar(200),
  date date,
  judge longtext,
  case_level varchar(200),
  prosecutor varchar(200),
  appellee varchar(200),
  case_type varchar(200),
  abstract LONGTEXT,
  prosecutor_detail varchar(2000),
  appellee_detail varchar(2000),
  url varchar(200),
  inquisition longtext,
  facts_and_evidence longtext,
  confession_of_defense longtext,
  advocate longtext,
  details longtext,
  judge_reason longtext,
  judgment_result longtext,
  judgment_people varchar(200),
  recoder varchar(1000),
  content longtext,
  criminal varchar(2000),
  keywords varchar(10000)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



create table doc2rule(
  _id int unsigned not null primary key auto_increment,
  doc_id int(11),
  rule_id int(11)
)ENGINE=InnoDB DEFAULT CHARSET = utf8;






















