import requests
from bs4 import BeautifulSoup
from lxml import etree
import csv

header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.41"}
#设置网站代理，反爬虫操作

f = open('C:\pycharm\chatting_Robot(python_class_design)\dataset.csv','w',encoding = "utf-8",newline = ''"")
#内容写入，避免空格出现
csv_writer = csv.writer(f)
csv_writer.writerow(["名称","网址"])

for i in range(1,108):
    url_1 = 'http://college.gaokao.com/schlist/p'
    url = url_1 + str(i) + '/'#更新网站，实现翻页操作
    response = requests.get(url,headers = header)
    soup = BeautifulSoup(response.text, 'lxml')

    grid = soup.find_all(name="strong", attrs={"class": "blue"})
    for word in grid:
        lst = word.find(name="a")#解析网页信息，查询定位信息
        csv_writer.writerow([lst.string,lst['href']])
        print(lst.string)
        print(lst['href'])