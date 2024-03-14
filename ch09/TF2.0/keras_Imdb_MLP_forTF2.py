#!/usr/bin/env python
# coding: utf-8



# # 数据准备




# 导入所需的模块
import urllib.request
import os
import tarfile
################下载IMDB数据集###########################
#下载地址
url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#设置存储文件的路径
filepath="aclImdb_v1.tar.gz"
#判断文件不存在就会下载文件
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
#判断解压缩目录是否存在，打开压缩文件，解压缩到相应目录
if not os.path.exists("./aclImdb"):
    tfile = tarfile.open("./aclImdb_v1.tar.gz", 'r:gz')
    result = tfile.extractall('imdb/')


from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
tf.compat.v1.disable_eager_execution()




import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)





import os
def read_files(filetype):
    path = "imdb/aclImdb/"
    file_list=[]

    positive_path=path + filetype+"/pos/"
    for f in os.listdir(positive_path):
        file_list+=[positive_path+f]
    
    negative_path=path + filetype+"/neg/"
    for f in os.listdir(negative_path):
        file_list+=[negative_path+f]
        
    print('read',filetype, 'files:',len(file_list))
       
    all_labels = ([1] * 12500 + [0] * 12500) 
    
    all_texts  = []
    
    for fi in file_list:
        with open(fi,encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
            
    return all_labels,all_texts




y_train,train_text=read_files("train")


# In[16]:


y_test,test_text=read_files("test")





token = Tokenizer(num_words=2000)
token.fit_on_texts(train_text)



x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)





x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test  = sequence.pad_sequences(x_test_seq,  maxlen=100)


# # 建立模型



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten
from tensorflow.keras.layers import Embedding
#from keras.layers import Dense, Dropout, Activation,Flatten
#from keras.layers import Embedding

model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=2000, 
                    input_length=100))
model.add(Dropout(0.2))


# In[24]:


model.add(Flatten())
model.add(Dense(units=256,activation='relu' ))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid' ))


# In[25]:


print(model.summary())



# # 训练模型


model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
train_history =model.fit(x_train, y_train,batch_size=100, 
                         epochs=10,verbose=2,
                         validation_split=0.2)


scores=model.evaluate(x_test,y_test,verbose=1)
print(scores[1])



import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'accuracy','val_accuracy')

show_train_history(train_history,'loss','val_loss')


# # 评估模型的准确率


scores = model.evaluate(x_test, y_test, verbose=1)
scores[1]


# # 预测概率




probility=model.predict(x_test)





probility[:10]





for p in probility[12500:12510]:
    print(p)


# # 预测结果




predict=model.predict(x_test)





predict[:10]





predict_classes=predict.reshape(-1)
predict_classes[:10]


# # 查看预测结果




SentimentDict={1:'正面的',0:'负面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('标签label:',SentimentDict[y_test[i]],
          '预测结果:',SentimentDict[predict_classes[i]])







#预测新的影评





input_text='''
Oh dear, oh dear, oh dear: where should I start folks. I had low expectations already because I hated each and every single trailer so far, but boy did Disney make a blunder here. I'm sure the film will still make a billion dollars - hey: if Transformers 11 can do it, why not Belle? - but this film kills every subtle beautiful little thing that had made the original special, and it does so already in the very early stages. It's like the dinosaur stampede scene in Jackson's King Kong: only with even worse CGI (and, well, kitchen devices instead of dinos).
The worst sin, though, is that everything (and I mean really EVERYTHING) looks fake. What's the point of making a live-action version of a beloved cartoon if you make every prop look like a prop? I know it's a fairy tale for kids, but even Belle's village looks like it had only recently been put there by a subpar production designer trying to copy the images from the cartoon. There is not a hint of authenticity here. Unlike in Jungle Book, where we got great looking CGI, this really is the by-the-numbers version and corporate filmmaking at its worst. Of course it's not really a "bad" film; those 200 million blockbusters rarely are (this isn't 'The Room' after all), but it's so infuriatingly generic and dull - and it didn't have to be. In the hands of a great director the potential for this film would have been huge.
Oh and one more thing: bad CGI wolves (who actually look even worse than the ones in Twilight) is one thing, and the kids probably won't care. But making one of the two lead characters - Beast - look equally bad is simply unforgivably stupid. No wonder Emma Watson seems to phone it in: she apparently had to act against an guy with a green-screen in the place where his face should have been. 
'''



input_seq = token.texts_to_sequences([input_text])


len(input_seq[0])



pad_input_seq  = sequence.pad_sequences(input_seq , maxlen=100)





len(pad_input_seq[0])





predict_result=model.predict(pad_input_seq)


predict_result[0][0]




# # serialize model to JSON




model_json = model.to_json()
with open("Imdb_mlp_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("Imdb_mlp_model.h5")
print("Saved model to disk")







