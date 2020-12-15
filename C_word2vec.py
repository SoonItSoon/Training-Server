import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

train_data = pd.read_csv('data.csv', encoding='utf-8')

print(len(train_data))
print(train_data.isnull().values.any())
train_data['내용'] = train_data['내용'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 1-9]","")
print(train_data[:5])
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다', '바랍니다', '습니다', '하십시오']
okt = Okt()
tokenized_data = []

for sentence in train_data['내용']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    tokenized_data.append(temp_X)

print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

model = Word2Vec(sentences = tokenized_data, size = 100, window = 7, min_count = 5, workers = 4, sg = 1)
# 0: CBOW, 1: Skip-gram(sg) 벡터 크기(size) 고려할 앞 뒤 단어 개수(window), 최소 등장횟수(min_count), 코어 수(workers)

model.wv.vectors.shape

print(model.wv.most_similar("코로나"))
print(model.wv.most_similar("지진"))

model.wv.save_word2vec_format('word2vec.bin', binary=True)

