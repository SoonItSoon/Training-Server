from krwordrank.word import KRWordRank
import pandas as pd

dataset_train1 = pd.read_csv("./COVID-nouns.csv", encoding="utf-8")
# dataset_train1.drop(["SUB"], axis=1, inplace=True)
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "확진자발생"]
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "보건소방문"]
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "안내"]
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "안내"].loc[dataset_train1["SUB"] == "개인방역수칙"]
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "안내"].loc[dataset_train1["SUB"] == "발생현황"]
data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "안내"].loc[dataset_train1["SUB"] == "행정안내"]
new_df = pd.DataFrame(data1)
new_df = new_df["MESSAGE"]
print(new_df.head())
texts = new_df.values.tolist()

min_count = 5  # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 10  # 단어의 최대 길이
wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

beta = 0.85  # PageRank의 decaying factor beta
max_iter = 10
# texts = ['예시 문장 입니다', '여러 문장의 list of str 입니다', ...]
keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:30]:
    print(f"{word:12s}\t: {r:.4f}")

