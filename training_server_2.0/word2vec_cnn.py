import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Input, Flatten, Concatenate
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import classification_report

plt.draw()
train_data = pd.read_csv("./csv/alertMsg_train_top10.csv", names=["MESSAGE", "CATEGORY"], encoding='utf-8')
train_data.message = train_data.MESSAGE.astype(str)

train_b = train_data["MESSAGE"][1:]
train_a = train_data["CATEGORY"][1:]

intent_train = train_a.values.tolist()
label_train = train_b.values.tolist()

test_data = pd.read_csv("./csv/alertMsg_test_top10.csv", names=["MESSAGE", "CATEGORY"], encoding='utf-8')
test_data.message = test_data.MESSAGE.astype(str)

test_b = test_data["MESSAGE"][1:]
test_a = test_data["CATEGORY"][1:]

intent_test = test_a.values.tolist()
label_test = test_b.values.tolist()

# print(intent_test)
print('훈련용 문장의 수 :', len(intent_train))
print('훈련용 레이블의 수 :', len(label_train))
print('테스트용 문장의 수 :', len(intent_test))
print('테스트용 레이블의 수 :', len(label_test))

idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(label_train)

label_train = idx_encode.transform(label_train)
label_test = idx_encode.transform(label_test)

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
print(label_idx)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(intent_train)
sequences = tokenizer.texts_to_sequences(intent_train)
print(sequences[:5])

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
print('단어 집합(Vocabulary)의 크기 :', vocab_size)

print('문장의 최대 길이 :', max(len(l) for l in sequences))
print('문장의 평균 길이 :', sum(map(len, sequences)) / len(sequences))
plt.hist([len(s) for s in sequences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 70
intent_train = pad_sequences(sequences, maxlen=max_len)
label_train = to_categorical(np.asarray(label_train))
print('전체 데이터의 크기(shape):', intent_train.shape)
print('레이블 데이터의 크기(shape):', label_train.shape)

print(intent_train[1])
print(label_train[1])

indices = np.arange(intent_train.shape[0])
np.random.shuffle(indices)
print(indices)

intent_train = intent_train[indices]
label_train = label_train[indices]

n_of_val = int(0.1 * intent_train.shape[0])
print(n_of_val)

X_train = intent_train[:-n_of_val]
y_train = label_train[:-n_of_val]
X_val = intent_train[-n_of_val:]
y_val = label_train[-n_of_val:]
X_test = intent_test
y_test = label_test

print('훈련 데이터의 크기(shape):', X_train.shape)
print('검증 데이터의 크기(shape):', X_val.shape)
print('훈련 데이터 레이블의 개수(shape):', y_train.shape)
print('검증 데이터 레이블의 개수(shape):', y_val.shape)
print('테스트 데이터의 개수 :', len(X_test))
print('테스트 데이터 레이블의 개수 :', len(y_test))

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec/word2vec_all.bin", binary=True)
print(word2vec_model.vectors.shape)
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))
print(np.shape(embedding_matrix))


def get_vector(word):
    if word in word2vec_model:
        return word2vec_model[word]
    else:
        return None


for word, i in word_index.items():  # 훈련 데이터의 단어 집합에서 단어와 정수 인덱스를 1개씩 꺼내온다.
    temp = get_vector(word)  # 단어(key) 해당되는 임베딩 벡터의 300개의 값(value)를 임시 변수에 저장
    if temp is not None:  # 만약 None이 아니라면 임베딩 벡터의 값을 리턴받은 것이므로
        embedding_matrix[i] = temp  # 해당 단어 위치의 행에 벡터의 값을 저장한다.

filter_sizes = [2, 3, 5]  # 사용하는 커널의 사이즈
num_filters = 512  # 커널들의 개수
drop = 0.5  # 드롭아웃 확률 50%

model_input = Input(shape=(max_len,))
z = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
              input_length=max_len, trainable=False)(model_input)

conv_blocks = []

for sz in filter_sizes:
    conv = Conv1D(filters=num_filters,
                  kernel_size=sz,
                  padding="valid",
                  activation="relu",
                  strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(drop)(z)
model_output = Dense(len(label_idx), activation='softmax')(z)

model = Model(model_input, model_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())
history = model.fit(X_train, y_train,
                    batch_size=100,  # 1회 학습시 주는 데이터 개수
                    epochs=20,  # 전체 데이타 10번 학습
                    validation_data=(X_val, y_val))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_len)

y_predicted = model.predict(X_test)
y_predicted = y_predicted.argmax(axis=-1)  # 예측된 정수 시퀀스로 변환

y_predicted = idx_encode.inverse_transform(y_predicted)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환
y_test = idx_encode.inverse_transform(y_test)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환

print('accuracy: ', sum(y_predicted == y_test) / len(y_test))
print("Precision, Recall and F1-Score:\n\n", classification_report(y_test, y_predicted))

tf.keras.models.save_model(model, 'mnist_mlp_model.h5')
