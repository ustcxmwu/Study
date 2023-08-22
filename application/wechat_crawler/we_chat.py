#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   we_chat.py
@Time    :   2023-04-03 10:32
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pprint
import time
import requests
import csv

f = open('青灯公众号文章.csv', mode='a', encoding='utf-8', newline='')
csv_writer = csv.DictWriter(f, fieldnames=['标题', '文章发布时间', '文章地址'])
csv_writer.writeheader()


def parser():
    for page in range(0, 40, 5):
        url = f'https://mp.weixin.qq.com/cgi-bin/appmsg?action=list_ex&begin=0&count=5&fakeid=MzUyNjQxNjxxxx==&type=9' \
              f'&query=&token=80433369&lang=zh_CN&f=json&ajax=1'
        headers = {
            'cookie': 'appmsglist_action_3946468299=card; rewardsn=; wxtokenkey=777; wwapp.vid=; wwapp.cst=; wwapp.deviceid=; pgv_info=ssid=s8444341482; pgv_pvid=3244579096; ua_id=ga2hywMB64VrbDEaAAAAANL06yfV91KOnE1gY1YD1Hs=; wxuin=80486703229906; uuid=b231ed3d4f4046d115c78a2986bc72d6; cert=q2MGE5Sc3PLe4ZjS4M5zDab_ViTNVDjX; sig=h015ba0c82744067e81ef2c5d28d54ad514a346eee19edadbfbe5a30c3850a57c6213fd293a7db323c0; data_bizuin=3946468299; bizuin=3946468299; master_user=gh_1d55abeddca5; master_sid=c3ZpczlTc0lQcWVZbDFwcm52OWNPVW14U1h1RHNqV0RoRFVxVVpJMWtuclJIQlRjN3RYT0pKSl9rTFlmQm5YaF9leDJQR2lsQ0EybERaVjFRNmdzVVEzeHlwdldFYUk5WWJIVUlKVjI0dkkweUpzbFVrUHZsN0VoWG1pYXYxV1A1WTdwM1ZQWFlyazR2TllC; master_ticket=78c6efebf45352d0802d8892241175fa; media_ticket=acf6f1cf9f5ffd3dfe3350395b4798ea075aa53a; media_ticket_id=3946468299; data_ticket=culBj/r3W+0Cxq05vqOD2/eJduuNceXCDRtX6miz6hQUdYVXev5UFcE2g718WtTC; rand_info=CAESIHqE9qiBPqvo8VARJc4E+U9NNAKgPBrkpfRILgQUFkXB; slave_bizuin=3946468299; slave_user=gh_1d55abeddca5; slave_sid=Z1RqdXJpdU5BMDA0WW52TU0xVjNhbjBtckpkUEV3ZWl5aEpXS0FFcU1ubzBicnVUUkZSMDJTekZZeXh4NGFvZmdrVmk4M0dUM1FiT3NtYXdTdnR3X1JxZ2VHZW1ONU8wd1lRRk5hUEZ4YVRna1FuMGhpMGFRZHA1Z3l4dDFSZE5GSU1HdFJCQmlLWWdna3Fz; _clck=3946468299|1|fag|0; _clsk=eb3jp7|1680489265811|5|1|mp.weixin.qq.com/weheat-agent/payload/record',
            'referer': 'https://mp.weixin.qq.com/cgi-bin/appmsg?t=media/appmsg_edit&action=edit&type=77&appmsgid=100000001&token=80433369&lang=zh_CN',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
        }

        response = requests.get(url=url, headers=headers)
        html_data = response.json()
        pprint.pprint(response.json())
        lis = html_data['app_msg_list']
        for li in lis:
            title = li['title']
            link_url = li['link']
            update_time = li['update_time']
            timeArray = time.localtime(int(update_time))
            otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
            dit = {
                '标题': title,
                '文章发布时间': otherStyleTime,
                '文章地址': link_url,
            }
            csv_writer.writerow(dit)
            print(dit)


if __name__ == '__main__':
    parser()