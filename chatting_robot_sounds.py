#程序模块引入
import csv
from fuzzywuzzy import fuzz
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
import pyttsx3
import wave
import time
import pyaudio
from aip import AipSpeech
from selenium import webdriver

#声音录制设置
CHUNK = 1024
FORMAT = pyaudio.paInt16 # 16位深
CHANNELS = 1 #1是单声道，2是双声道。
RATE = 16000 # 采样率，调用API一般为8000或16000
RECORD_SECONDS = 10 # 录制时间10s

#录音文件保存路径
def save_wave_file(pa, filepath, data):
    wf = wave.open(filepath, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pa.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(data))
    wf.close()

#录音主体文件
def get_audio(filepath,isstart):
    '''
    :param filepath:文件存储路径（'test.wav'） 
    :param isstart: 录音启动开关（0：关闭 1：开启）
    '''
    if isstart == 1:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         frames_per_buffer=CHUNK)

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)  # 读取chunk个字节 保存到data中
            frames.append(data)  # 向列表frames中添加数据data

        stream.stop_stream()
        stream.close()  # 停止数据流
        pa.terminate()  # 关闭PyAudio

        #写入录音文件
        save_wave_file(pa, filepath, frames)
    elif isstart == 0:
        exit()

#获得录音文件内容
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

#百度语音识别编码
APP_ID = '24220984'
API_KEY = '2Qc81SISAOn9uQfvzbf3hgo4'
SECRET_KEY = 'Vn7OV7YukAmhWtHX9d333FAGbaDmvpHn'

#语音模块主体函数
def speech_record(isstart):
    '''
    :param isstart: 录音启动开关（0：关闭 1：开启）
    :return: sign:是否获取到声音信号（0：未获取 1：获取到） result_out:返回识别的语义信息(none为未获取到语音信息)
    '''
    sign =1
    result_out = ""
    filepath = 'test.wav'
    get_audio(filepath,isstart)
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    result = client.asr(get_file_content('test.wav'), 'wav',16000,{'dev_pid': 1537,})
    if 'result' not in result.keys():
        sign = 0
        result_out = None
    elif result['result'] == ['']:
        sign = 0
        result_out = None
    else:
        result_out = "".join(result['result'])
    return [sign,result_out]


#语音播报函数
def speech_read(result):
    #模块初始化
    engine = pyttsx3.init()
    #print('准备开始语音播报...')
    engine.say(result)
    # 等待语音播报完毕
    engine.runAndWait()

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



#程序运行主函数
if __name__ == '__main__':

    #贝叶斯分类器训练
    train_data_apply = jieba_text(train_data)[1]
    train_label_apply = jieba_text(train_label)[0]
    train_array = tf_idf(train_data_apply,['你好'])[0]
    model_apply = bayes_model(train_array,train_label_apply)

    #起始播报
    greeting1 = '你好，我是智能机器人zz,我可以陪你聊天，也可以帮助您查询院校信息'
    greeting2 = '让我们开始聊天吧'
    speech_read(greeting1)
    speech_read(greeting2)

    #主循环
    while True:
        a = speech_record(1)
        print(a)
        if a[0] == 0:
            sph = '你还在吗？我走啦'
            speech_read(sph)
            speech_record(0)
            break
        elif a[0] == 1:
            similarity = fuzz.ratio(a[1], "拜拜")
            if similarity > 50:
                reply = '今天和你聊的很开心，再见啦，拜拜'
                speech_read(reply)
                speech_record(0)
                break
            else:
                input_list = jieba_text(a[1])[1]
                input_apply = tf_idf(train_data_apply,input_list)[1]
                value = model_apply.predict(input_apply)

                if value == [' A ']:
                    reply = get_greeting(a[1],question_greeting,answer_greeting)
                    if reply == None:
                        reply_none = 'zz才疏学浅，对主人的问题无能为力'
                        speech_read(reply_none)
                    else:
                        speech_read(str(reply))
                if value == [' B ']:
                    reply = get_greeting(a[1],question_dataset,answer_dataset)
                    if reply == None:
                        reply_none = 'zz才疏学浅，对主人的问题无能为力'
                        speech_read(reply_none)
                    else:
                        speech_read('我为你找到以下内容,你可以在本网页上查询院校信息')
                        web_open(str(reply))
                        break






















