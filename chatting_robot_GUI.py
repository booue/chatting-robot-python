#gui模块引入
from PyQt5 import QtWidgets  # 导入相关组件
import robot  # 导入登录界面的py文件
#主程序模块引入
import csv
from fuzzywuzzy import fuzz
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
from selenium import webdriver


#娱乐聊天数据库读取
question_greeting = []
answer_greeting = []
with open("greeting.csv", 'r',encoding='GBK') as f:
    greeting = csv.reader(f)
    header = next(greeting)
    for words in greeting:
        question_greeting.append(words[0])
        answer_greeting.append(words[1])

#功能查询模块读取
question_dataset = []
answer_dataset = []
with open("dataset.csv", 'r', encoding='GBK') as f:
    dataset = csv.reader(f)
    header = next(dataset)
    for words in dataset:
        question_dataset.append(words[0])
        answer_dataset.append(words[1])

#模糊匹配功能实现
def get_greeting(input_questions,question,answer):
    text = {}
    for key, value in enumerate(question):
        similarity = fuzz.ratio(input_questions, value)
        if similarity > 20:
            text[key] = similarity
    #text中存储的为匹配程度较高（大于60）的问题以及相应的键值（key）
    if len(text) > 0:
        train = sorted(text.items(), key=lambda d: d[1], reverse=True)
        #按照字典键值（value = 匹配值）进行排序
        answer3 = answer[train[0][0]]
        #取出匹配程度最高的问题对应索引
    else:
        answer3 = None
    return  answer3

#训练文件内容读取
train_data = ""
train_label = ""
with open("train_data.csv", 'r',encoding='GBK') as f:
    train= csv.reader(f)
    header = next(train)
    for words in train:
        train_data += words[0]+'\n'
        train_label += words[1]+'\n'
train_data = train_data.strip('\n')
train_label = train_label.strip('\n')

# 停用词文件读取
stpwrdpath = "C:\python_learning\pycharm\chatting_Robot(python_class_design)\stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'r',encoding = 'utf-8')
stpwrd_content = stpwrd_dic.read()

#样本格式化处理：（空格分隔，字符串转换，去除停用词）
def jieba_text(text):
    '''
    :param text:待处理文本 
    :return: text_list：分隔后文本 text_apply:分隔处理，去除停用词操作
    '''
    text_jieba = jieba.cut(text)
    text_str = ""
    for word in text_jieba:
        text_str +=word+" "
    text_list = text_str.split('\n')#空格分隔，换行符添加
    text_apply = []
    for file in text_list:
        for word in file:
            if word in stpwrd_content:
                file = file.replace(word,'')#停用词替换
            else:
                continue
        text_apply.append(file)
    return [text_list,text_apply]

#tf-idf数学处理
def tf_idf(text_train,text_test):
    '''
    :param text_train:训练样本处理 
    :param text_test: 预测样本处理
    :return: train_array:处理后训练样本 test_array:处理后预测样本
    '''
    vectorizer = CountVectorizer(min_df=1,max_features= 6)
    transformer = TfidfTransformer()
    tfidf_train = transformer.fit_transform(vectorizer.fit_transform(text_train))
    tfidf_test = transformer.fit_transform(vectorizer.transform(text_test))
    #训练样本和预测样本在相同模型处理，保证维度一致

    train_array = tfidf_train.toarray()
    test_array = tfidf_test.toarray()
    #稀疏矩阵转换为稠密矩阵，保证维度一致

    return [train_array,test_array]

#贝叶斯模型训练
def bayes_model(dataset,label):
    '''
    :param dataset:训练样本 
    :param label: 样本标签
    :return: 训练好的model样本
    '''
    model = MultinomialNB()
    model.fit(dataset, label)
    return model

#网页自动化函数
def web_open(result):
    '''
    :param result: 搜索框输入内容 
    '''
    driver = webdriver.Firefox()
    driver.get(result)


#贝叶斯分类器训练
train_data_apply = jieba_text(train_data)[1]
train_label_apply = jieba_text(train_label)[0]
train_array = tf_idf(train_data_apply,['你好'])[0]
model_apply = bayes_model(train_array,train_label_apply)

def on_click(self):
    text_1 = ui.textEdit.toPlainText()  # 用户输入
    ui.textBrowser_2.setText(text_1)
    input_list = jieba_text(text_1)[1]
    input_apply = tf_idf(train_data_apply, input_list)[1]
    value = model_apply.predict(input_apply)

    if value == [' A ']:
        reply = get_greeting(text_1, question_greeting, answer_greeting)
        if reply == None:
            reply_none = 'zz才疏学浅，对主人的问题无能为力'
            ui.textBrowser.setText(reply_none)
        else:
            ui.textBrowser.setText(str(reply))
    if value == [' B ']:
        reply = get_greeting(text_1, question_dataset, answer_dataset)
        if reply == None:
            reply_none = 'zz才疏学浅，对主人的问题无能为力'
            ui.textBrowser.setText(reply_none)
        else:
            ui.textBrowser.setText('我为你找到以下内容,你可以在本网页上查询院校信息')
            web_open(str(reply))

def off_click(self):
    ui.textEdit.clear()#清除聊天框内容的操作

app = QtWidgets.QApplication([])
window = QtWidgets.QMainWindow()
ui = robot.Ui_MainWindow()
ui.setupUi(window)  # 启动运行
ui.pushButton.clicked.connect(on_click)
ui.pushButton_2.clicked.connect(off_click)
window.show()  # 显示窗口
app.exec()


























