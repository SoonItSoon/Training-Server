import gensim
from eunjeon import Mecab
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import warnings
from sklearn.manifold import TSNE
import os.path
import pickle

USE_PREV_MODEL = True
warnings.filterwarnings("ignore")
mecab = Mecab()


def tokenize_sentense(text):
    noun_list = mecab.nouns(text)
    final_list = []
    for noun in noun_list:
        if len(noun) >= 2:
            final_list.append(noun)
    return final_list


def get_sentence_mean_vector(morphs):
    vector = []
    for i in morphs:
        try:
            vector.append(w2v_model.wv[i])
        except KeyError as e:
            pass
    try:
        return np.mean(vector, axis=0)
    except IndexError as e:
        pass


dataset = pd.read_csv("../csv/AM_210329_COVID9.csv", encoding="utf-8")
message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
tokenized_data = []

if USE_PREV_MODEL:
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec.bin")
    print("Word2Vec 모델 로드 완료")
else:
    for message in message_list:
        message_split = message.split("]")
        new_message = message_split[1]
        if len(message_split) > 2:
            for message_piece in message_split[2:]:
                new_message = f"{new_message}]{message_piece}"
        tokenized_data.append(tokenize_sentense(new_message))
    print("Word2Vec 모델 생성 중...")
    w2v_model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=10, iter=20, sg=0, workers=8)
    w2v_model.wv.save_word2vec_format("word2vec.bin")
    print("Word2Vec 모델 생성 및 저장 완료")

# model_result = w2v_model.wv.most_similar("발생")
# print(model_result)

dataset["mecab_message"] = dataset["MESSAGE"].map(tokenize_sentense)
print("재난문자 데이터 명사 토큰화 완료")
dataset["mecab_len_message"] = dataset["mecab_message"].map(len)
dataset["wv"] = dataset["mecab_message"].map(get_sentence_mean_vector)
print("재난문자 데이터 벡터화 완료")
# print(dataset.head())

num_clusters = 3
word_vectors = dataset.wv.to_list()

kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)
dataset["CLASSIFICATION"] = idx

print(dataset.CATEGORY.value_counts())
print(dataset.CLASSIFICATION.value_counts())
# print(dataset.head())
# print(len(dataset))
# print(dataset.loc[dataset["CLASSIFICATION"] == 0][["CATEGORY", "CLASSIFICATION", "MESSAGE"]].head())
# con_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
con_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(0, len(dataset)):
    con_matrix[dataset["CATEGORY"][i]][dataset["CLASSIFICATION"][i]] += 1

for i in range(0, 3):
    print(con_matrix[i])

# # 클러스터 시각화
# X = dataset["wv"].to_list()
# y = dataset["CLASSIFICATION"].to_list()
#
# tsne_filepath = "tsne3000.pkl"
#
# # File Cache
# if not os.path.exists(tsne_filepath):
#     tsne = TSNE(random_state=42)
#     tsne_points = tsne.fit_transform(X)
#     with open(tsne_filepath, 'wb+') as f:
#         pickle.dump(tsne_points, f)
# else:  # Cache Hits!
#     with open(tsne_filepath, 'rb') as f:
#         tsne_points = pickle.load(f)
#
# tsne_df = pd.DataFrame(tsne_points, index=range(len(X)), columns=["x_coord", "y_coord"])
# tsne_df["user_bio"] = dataset["MESSAGE"].to_list()
# tsne_df["cluster_no"] = y
# print(tsne_df.sample(3))

# ALL
# con_matrix = [[4764, 388, 627, 4168, 53],
#               [5, 7578, 101, 236, 2],
#               [1, 117, 2781, 44, 229],
#               [1, 81, 2221, 97, 57],
#               [3, 790, 492, 39, 2019]]
# MAIN
# con_matrix = [[8356, 774, 870],
#               [96, 7729, 97],
#               [70, 1116, 7796]]
# SUB
con_matrix = [[2723, 272, 177],
              [2180, 225, 52],
              [418, 1037, 1888]]
accuracy = 0
for i in range(0, 3):
    accuracy += con_matrix[i][i]
accuracy /= len(dataset)
precision = 0
recall = 0
f1_score = 0
for i in range(0, 3):
    print(con_matrix[i])
    TP = con_matrix[i][i]
    FP = 0
    FN = 0
    for j in range(0, 3):
        if j != i:
            FP += con_matrix[j][i]
            FN += con_matrix[i][j]
    precision += (TP / (TP + FP))
    recall += (TP / (TP + FN))
precision /= 3
recall /= 3
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Accuracy : {accuracy:.6f}\nF1-Score : {f1_score:.6f}(Precision : {precision:.6f} Recall : {recall:.6f})")
