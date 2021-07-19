import gensim
from eunjeon import Mecab
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import pandas as pd
import warnings

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

# {"보건소방문": 0, "확진자발생": 1, "발생현황": 2, "행정안내": 3, "개인방역수칙": 4}
def get_n_tokens(data, model):
    result = []
    word_list = ["확진", "보건소", "착용", "자제", "운영", "발생", "방문", "동선", "검사", "지역"]
    for i, j in enumerate(data):
        dist = [0] * len(word_list)
        if not data[i]:  # 내용이 없는 글
            pass
        elif len(data[i]) == 1:  # 한 토큰으로 된 글
            for l, m in enumerate(word_list):
                try:
                    dist[l] = model.similarity(m, data[0])
                except:
                    continue
        else:  # 한 토큰 이상으로 된 글
            for idx, k in enumerate(j):
                if idx >= 50:  # 토큰 50개만 사용한다.
                    break
                for l, m in enumerate(word_list):
                    try:
                        dist[l] += model.similarity(m, k)
                    except:
                        continue
        result.append([x / (idx + 1) for x in dist])
    return result


def unsupervised_learning(data, test_data, n, answer_dict):
    km_model = KMeans(n_clusters=n, algorithm='auto')
    km_model.fit(data)  # 학습
    predict_list = km_model.predict(data)  # clustering
    group_list = [[] for _ in range(n)]  # 잘 됐는지 확인하려는 list
    con_matrix = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    for i in range(n):
        for idx, j in enumerate(predict_list):
            if j == i:
                group_list[i].append(test_data[idx])
                con_matrix[answer_dict[category_list[idx]]][i] += 1
    return predict_list, group_list, con_matrix


dataset = pd.read_csv("../csv/AM_210329_COVID7.csv", encoding="utf-8")
message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
tokenized_data = []

for message in message_list:
    message_split = message.split("]")
    new_message = message_split[1]
    if len(message_split) > 2:
        for message_piece in message_split[2:]:
            new_message = f"{new_message}]{message_piece}"
    tokenized_data.append(tokenize_sentense(new_message))
print("토큰화 완료")
if USE_PREV_MODEL:
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("word2vec.bin")
    print("Word2Vec 모델 로드 완료")
else:
    print("Word2Vec 모델 생성 중...")
    w2v_model = Word2Vec(tokenized_data, size=100, window=5, min_count=10, iter=20, sg=1, workers=8)
    print("Word2Vec 모델 생성 완료")
    w2v_model.wv.save_word2vec_format("word2vec.bin")

test_list = ["[진도군청]오늘(7.1.)부터 코로나19 행정명령을 발령하오니 대중교통 이용시 마스크 착용을 의무화합니다. 마스크 미착용시 행정처분이 될 수 있으니 협조바랍니다.",
             "[금산군청] 자가격리중인 금산 1번 확진자의 접촉자 가족3명은 진단검사 결과 모두 음성으로 나왔습니다. 마스크 착용 등 방역수칙을 준수해 주시기 바랍니다.",
             "[장흥군청] 씨씨씨아가페실버센터(6.26~29), 광주사랑교회(6.28) 등 방문자는 외출을 자제하시고, 장흥군보건소(061-860-6481)로 신고바랍니다."]
test_data = []
for message in test_list:
    test_data.append(tokenize_sentense(message))
print("테스트 데이터 토큰화 완료")
vector_data = get_n_tokens(tokenized_data, w2v_model)
# print(vector_data[0])
print("벡터 계산 완료")
inertia = []  # cluster 응집도

# k = 5로 결정
# for k in range(1, 11):  # 10개까지
#     km_model = KMeans(n_clusters=k)
#     km_model.fit(vector_data)
#     inertia.append(km_model.inertia_)
# print(inertia)
num_clusters = 5

# {0: "보건소방문", 1: "확진자발생", 2: "발생현황", 3: "행정안내", 4: "개인방역수칙"}
# answer_dict = {"보건소방문": 0, "확진자발생": 1, "발생현황": 2, "행정안내": 3, "개인방역수칙": 4}
# answer_dict = {0: 2, 1: 1, 2: 0, 3: 4, 4: 3}
answer_dict = {0: 2, 1: 1, 2: 0, 3: 4, 4: 3}

predict_list, group_list, con_matrix = unsupervised_learning(vector_data, message_list, num_clusters, answer_dict)

# print(predict_list)
print(len(message_list))
for i in range(0, 5):
    print(con_matrix[i])
for i in range(0, 5):
    print(len(group_list[i]))

idx = list(predict_list)
names = w2v_model.wv.index2word
word_centroid_map = {names[i]: idx[i] for i in range(len(names))}

for c in range(num_clusters):
    # 클러스터 번호를 출력
    print("\ncluster {}".format(c))

    words = []
    cluster_values = list(word_centroid_map.values())
    for i in range(len(cluster_values)):
        if cluster_values[i] == c:
            words.append(list(word_centroid_map.keys())[i])
    print(words)


# word_vectors = w2v_model.wv.syn0
# kmeans_clustering = KMeans(n_clusters=num_clusters)
# idx = kmeans_clustering.fit_predict(word_vectors)
#
# idx = list(idx)
# names = w2v_model.wv.index2word
# word_centroid_map = {names[i]: idx[i] for i in range(len(names))}
#
# for c in range(num_clusters):
#     # 클러스터 번호를 출력
#     print("\ncluster {}".format(c))
#
#     words = []
#     cluster_values = list(word_centroid_map.values())
#     for i in range(len(cluster_values)):
#         if (cluster_values[i] == c):
#             words.append(list(word_centroid_map.keys())[i])
#     print(words)

# ### pyplot 그려주기 ###
# from sklearn.manifold import TSNE
# import matplotlib.font_manager as fm
# import matplotlib.pyplot as plt
# import matplotlib
#
# path_font = "C:/Users/soonwook/AppData/Local/Microsoft/Windows/Fonts/SeoulHangangB.ttf"
# prop = fm.FontProperties(fname=path_font)
# matplotlib.rcParams["axes.unicode_minus"] = False
#
# vocab = list(w2v_model.wv.vocab)
# X = w2v_model[vocab]
#
# tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X)
#
# import pandas as pd
#
# df = pd.DataFrame(X_tsne, index=vocab, columns=["x", "y"])
#
# print(df.head())
#
# fig = plt.figure()
# fig.set_size_inches(40, 20)
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(df["x"], df["y"])
#
# for word, pos in list(df.iterrows()):
#     ax.annotate(word, pos, fontsize=12, fontproperties=prop)
# plt.show()
#
# w2v_model.wv.save_word2vec_format("word2vec.bin")

# from scipy.spatial import distance_matrix
#
# distance = distance_matrix(word_vectors, word_vectors)
#
# distance_df = pd.DataFrame(distance, columns=names, index=names)
#
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
#
# word_data = []
# for word in tokenized_data:
#     for argu in word:
#         word_data.append(argu)
#
# vec = CountVectorizer()
# X = vec.fit_transform(set(word_data))
#
# TDM_DF = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()).T
#
# TDM_DF = TDM_DF.sum(axis=1).to_frame()
# TDM_DF.rename(columns={0: "doc1"}, inplace=True)
# print(TDM_DF.head())
#
# idx_bool = TDM_DF["doc1"] >= 1
# TDM_DF[idx_bool] = 1
#
# TDM_matrix = TDM_DF.loc[names].values
# distance_df_matrix = distance_df
#
# # classification_word = ["확진자발생", "보건소방문", "개인방역수칙", "발생현황", "행정안내"]
# classification_word = ["발생", "보건소", "방역", "발생", "행정"]
# target_matrix = distance_df.loc[classification_word, :].values
#
# print(target_matrix)
#
# import numpy as np
#
# result = np.matmul(target_matrix, TDM_matrix)
# maxpool = np.argmax(result)
# classification_rlt = classification_word[maxpool]
#
# print("classification result: {}".format(classification_rlt))
