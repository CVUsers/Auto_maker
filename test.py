"""
爬虫爬取数据
"""
import os
import requests
import re
import time

def find_max_name(data):
    max = 0
    try:
        for i in os.listdir(data):
            if int(i.split('.')[0]) >= max:
                max = int(i.split('.')[0])
    except:
        pass
    return max

headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"}

def url_get():
    url = 'http://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + word + '&pn='
    t = 0
    while t < 300:
        Url = url + str(t)
        try:
            neirong = requests.get(Url, headers=headers,timeout=3)
            print('正在下载....')
        except BaseException:
            t = t + 60
            print('exception')
            continue
        else:
            neirong.encoding="utf-8"
            zh=neirong.text
            zhxe=r'"objURL":"(.*?)"'
            pip=re.findall(zhxe,zh)
            time.sleep(2)
            return pip
if __name__ == '__main__':
    word = input('搜索：')
    max_index = find_max_name('mix_img/')
    for i in url_get():
        max_index += 1
        try:
            picture = requests.get(i).content
            with open(f"mix_img/{max_index}.jpg", "wb") as file:#文件保存路径
                file.write(picture)
        except:
            pass