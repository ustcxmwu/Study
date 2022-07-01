# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 10:24
# @Author  : Xiaomin Wu
# @Email   : ustcxmwu@gmai.com
# @Desc    :

from sqlalchemy import Column, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 创建对象的基类:
from package_test.ray_test.ray_client import main2

Base = declarative_base()


# 定义User对象:
class User(Base):
    # 表的名字:
    __tablename__ = 'user'

    # 表的结构:
    id = Column(String(20), primary_key=True)
    name = Column(String(20))


# 初始化数据库连接:
engine = create_engine('mysql+mysqlconnector://wuxiaomin:wxm1309@localhost:3306/freedream')
# 创建DBSession类型:
DBSession = sessionmaker(bind=engine)


if __name__ == '__main__':
    # 创建session对象:
    # session = DBSession()
    # # 创建新User对象:
    # new_user = User(id='5', name='Bob')
    # # 添加到session:
    # session.add(new_user)
    # # 提交即保存到数据库:
    # session.commit()
    # # 关闭session:
    # session.close()
    # with DBSession() as session:
    #     new_user = User(id='7', name='Bob')
    #     session.add(new_user)
    #     session.commit()

    # with DBSession() as session:
    #     # rows = session.query(User).filter(User.id == "5").all()
    #     rows = session.query(User).filter(User.id == "6").one()
    #     print(type(rows))
    #     print(rows.id)
    #     session.delete(rows)
    #     session.commit()


    with DBSession() as session:
        # rows = session.query(User).filter(User.id == "5").all()
        row = session.query(User).filter(User.id == "7").one()
        print(type(row))
        row.name = "xiaomin"
        session.commit()

