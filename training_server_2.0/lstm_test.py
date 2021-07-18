from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn import preprocessing

vector_size = 768

train_dataset = pd.read_csv("csv/dataframe_kobert_train_data.csv", encoding="utf-8")
train_labelset = pd.read_csv("csv/dataframe_kobert_train_label.csv", encoding="utf-8")
test_dataset = pd.read_csv("csv/dataframe_kobert_test_data.csv", encoding="utf-8")
test_labelset = pd.read_csv("csv/dataframe_kobert_test_label.csv", encoding="utf-8")

shape = train_dataset.shape
train_data = []
train_label = []
for i in range(shape[0]):
    b_list = []
    for j in range(shape[1] - 1):
        b_list.append(float(train_dataset[str(j)][i]))
    train_data.append(b_list)
    train_label.append(int(train_labelset["0"][i]))

shape = test_dataset.shape
test_data = []
test_label = []
for i in range(shape[0]):
    b_list = []
    for j in range(shape[1] - 1):
        b_list.append(float(test_dataset[str(j)][i]))
    test_data.append(b_list)
    test_label.append(int(test_labelset["0"][i]))

train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
# print(f"({len(train_data)}, {len(train_data[0])})")
# print(f"({len(train_label)})")
train_data = np.array(train_data).reshape(-1, 1, 768)
train_label = np.array(train_label).reshape(-1, 10)
test_data = np.array(test_data).reshape(-1, 1, 768)
test_label = np.array(test_label).reshape(-1, 10)
print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

model = Sequential()
# model.add(Embedding(vocab_size, 100))
# model.add(InputLayer(input_shape=(768)))
model.add(LSTM(768, input_shape=(1, 768), dropout=0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
history = model.fit(train_data, train_label, epochs=30, callbacks=[es, mc], batch_size=60, validation_data=(test_data, test_label))
# history = model.fit(train_data, train_label, epochs=15, batch_size=60, validation_split=0.2)
epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['acc'])
plt.plot(epochs, history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# y_predicted = model.predict(test_data)
# y_predicted = y_predicted.argmax(axis=-1)  # 예측된 정수 시퀀스로 변환
#
# y_predicted = idx_encode.inverse_transform(y_predicted)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환
# y_test = idx_encode.inverse_transform(y_test)  # 정수 시퀀스를 레이블에 해당하는 텍스트 시퀀스로 변환
#
# print('accuracy: ', sum(y_predicted == y_test) / len(y_test))
# print("Precision, Recall and F1-Score:\n\n", classification_report(y_test, y_predicted))
#
# tf.keras.models.save_model(model, 'mnist_mlp_model.h5')