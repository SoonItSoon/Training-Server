import pandas as pd

dataset = pd.read_csv("../csv/AM_210329_COVID4.csv", encoding="utf-8")

message_list = dataset["MESSAGE"].values.tolist()
sub_list = dataset["SUB"].values.tolist()

MAX_LEN = 5
PERCENTAGE = 0.6

keyword_3_1 = ["착용", "자제", "코로나", "마스크", "방문", "수칙", "방역", "모임", "예방", "지역"]
keyword_3_2 = ["자제", "코로나", "지역", "방문", "착용", "확진", "발생", "모임", "감염", "마스크"]
keyword_3_3 = ["운영", "착용", "시설", "금지", "단계", "선별", "코로나", "사회", "방역", "두기"]


def getSubCategory(count):
    count3_1 = count["3-1"] / MAX_LEN
    count3_2 = count["3-2"] / MAX_LEN
    count3_3 = count["3-3"] / MAX_LEN
    sub_category = []
    if count3_1 >= PERCENTAGE:
        sub_category.append(1)
    if count3_2 >= PERCENTAGE:
        sub_category.append(2)
    if count3_3 >= PERCENTAGE:
        sub_category.append(3)
    if len(sub_category) == 1:
        return sub_category[0]
    elif len(sub_category) == 0:
        return 4
    else:
        for i in range(0, MAX_LEN):
            for j in sub_category:
                if j == 1 and keyword_3_1[i] in message:
                    return 1
                elif j == 2 and keyword_3_2[i] in message:
                    return 2
                elif j == 3 and keyword_3_3[i] in message:
                    return 3
    return 4


sub_dict = {"개인방역수칙": 1, "발생현황": 2, "행정안내": 3}

con_matrix_sub = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
accuracy_sub = 0
for i in range(0, len(message_list)):
    message = message_list[i]
    count = {"3-1": 0, "3-2": 0, "3-3": 0}
    for j in range(0, MAX_LEN):
        if keyword_3_1[j] in message:
            count["3-1"] += 1
        if keyword_3_2[j] in message:
            count["3-2"] += 1
        if keyword_3_3[j] in message:
            count["3-3"] += 1
    sub_category = getSubCategory(count)
    con_matrix_sub[sub_dict[sub_list[i]] - 1][sub_category - 1] += 1
    if sub_dict[sub_list[i]] == sub_category:
        accuracy_sub += 1

print(accuracy_sub, len(message_list))
accuracy_sub /= len(message_list)
precision_sub = 0
recall_sub = 0
f1_score_sub = 0
print(accuracy_sub)
for i in range(0, 3):
    print(con_matrix_sub[i])
    TP = con_matrix_sub[i][i]
    FP = 0
    FN = 0
    for j in range(0, 4):
        if j != i:
            FP += con_matrix_sub[j][i]
            FN += con_matrix_sub[i][j]
    precision_sub += (TP / (TP + FP))
    recall_sub += (TP / (TP + FN))
precision_sub /= 3
recall_sub /= 3
f1_score_sub = 2 * (precision_sub * recall_sub) / (precision_sub + recall_sub)

print(f"Accuracy : {accuracy_sub:.6f}\nPrecision : {precision_sub:.6f}\nRecall : {recall_sub:.6f}\nF1-Score : {f1_score_sub:.6f}")

