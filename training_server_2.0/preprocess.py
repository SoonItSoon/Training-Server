import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 학습용 데이터셋 불러오기
dataset_train1 = pd.read_csv("csv/alertMsg_all.csv", encoding="utf-8")
# print(dataset_train1.head())

# 데이터 전처리
# dataset_train1.drop(["MID", "SEND_TIME", "SEND_LOC", "SEND_PLATFORM", "DISASTER"], axis=1, inplace=True)
# dataset_train1.drop(["SUB"], axis=1, inplace=True)
# print(dataset_train1.head())

# 대표 분류명 추출 (이미 csv 파일에서 전처리했으므로 안해도 무관)
# data1 = dataset_train1.loc[dataset_train1["CATEGORY"] == "전염병"]
# data2 = dataset_train1.loc[dataset_train1["CATEGORY"] == "태풍"]
# data3 = dataset_train1.loc[dataset_train1["CATEGORY"] == "호우"]
# data4 = dataset_train1.loc[dataset_train1["CATEGORY"] == "한파"]
# data5 = dataset_train1.loc[dataset_train1["CATEGORY"] == "교통"]
# data6 = dataset_train1.loc[dataset_train1["CATEGORY"] == "홍수"]
# data7 = dataset_train1.loc[dataset_train1["CATEGORY"] == "대설"]
# data8 = dataset_train1.loc[dataset_train1["CATEGORY"] == "폭염"]
# data9 = dataset_train1.loc[dataset_train1["CATEGORY"] == "산사태"]
# data10 = dataset_train1.loc[dataset_train1["CATEGORY"] == "기타"]
# # new_data = data1.append([data2, data3], sort=False)
# # new_df = pd.DataFrame(new_data)
# new_df1 = pd.DataFrame(data1)
# new_df1 = new_df1.sample(frac=1).head(5000)
# new_data = data2.append([data3, data4, data5, data6, data7, data8, data9, data10], sort=False)
# new_df = pd.DataFrame(new_data)
# new_df = pd.concat([new_df1, new_df], ignore_index=True)
# new_df = new_df[["MESSAGE", "CATEGORY"]]
# # new_data = dataset_train1
# # print(new_data.head())
# new_df.to_csv("csv/alertMsg_top10_sample.csv", encoding="utf-8-sig")

new_df = pd.DataFrame(dataset_train1)

# 분류명 라벨링
# encoder = LabelEncoder()
# encoder.fit(new_df["CATEGORY"])
# new_df["CATEGORY"] = encoder.transform(new_df["CATEGORY"])
# # print(new_data.head())
#
# # 라벨링된 분류명 매핑
# # {0: '개인방역수칙', 1: '발생현황', 2: '보건소방문', 3: '행정안내', 4: '확진자발생'}
# mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
# print(mapping)

# 학습, 테스트 셋 분리
train, test = train_test_split(new_df, test_size=0.1, random_state=42)
print("train shape is:", len(train))
print(train.head())
print("test shape is:", len(test))
print(test.head())

train.to_csv("csv/alertMsg_train_top10.csv", encoding="utf-8-sig")
test.to_csv("csv/alertMsg_test_top10.csv", encoding="utf-8-sig")

