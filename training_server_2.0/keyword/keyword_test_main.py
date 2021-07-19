import pandas as pd

dataset = pd.read_csv("../csv/AM_210329_COVID3.csv", encoding="utf-8")

message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()

MAX_LEN = 5
PERCENTAGE = 0.4

keyword_1 = ["확진", "발생", "동선", "역학", "접촉", "코로나", "공개", "홈페이지", "방역", "예정"]
keyword_2 = ["보건소", "방문", "검사", "선별", "진료소", "확진", "이용", "상담", "코로나", "교회"]
keyword_3 = ["착용", "자제", "코로나", "방문", "마스크", "지역", "모임", "방역", "수칙", "시설"]


def getMainCategory(count, message):
    count1 = count["1"] / MAX_LEN
    count2 = count["2"] / MAX_LEN
    count3 = count["3"] / MAX_LEN
    main_category = []
    if count1 >= PERCENTAGE:
        main_category.append(1)
    if count2 >= PERCENTAGE:
        main_category.append(2)
    if count3 >= PERCENTAGE:
        main_category.append(3)
    if len(main_category) == 1:
        return main_category[0]
    elif len(main_category) == 0:
        return 4
    else:
        for i in range(0, MAX_LEN):
            for j in main_category:
                if j == 1 and keyword_1[i] in message:
                    return 1
                elif j == 2 and keyword_2[i] in message:
                    return 2
                elif j == 3 and keyword_3[i] in message:
                    return 3
    return 4


main_dict = {"확진자발생": 1, "보건소방문": 2, "안내": 3}

con_matrix_main = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
accuracy_main = 0
for i in range(0, len(message_list)):
    message = message_list[i]
    count = {"1": 0, "2": 0, "3": 0}
    for j in range(0, MAX_LEN):
        if keyword_1[j] in message:
            count["1"] += 1
        if keyword_2[j] in message:
            count["2"] += 1
        if keyword_3[j] in message:
            count["3"] += 1
    main_category = getMainCategory(count, message)
    con_matrix_main[main_dict[category_list[i]] - 1][main_category - 1] += 1
    if main_dict[category_list[i]] == main_category:
        accuracy_main += 1

print(accuracy_main, len(message_list))
accuracy_main /= len(message_list)
precision_main = 0
recall_main = 0
f1_score_main = 0
print(accuracy_main)
for i in range(0, 3):
    print(con_matrix_main[i])
    TP = con_matrix_main[i][i]
    FP = 0
    FN = 0
    for j in range(0, 4):
        if j != i:
            FP += con_matrix_main[j][i]
            FN += con_matrix_main[i][j]
    precision_main += (TP / (TP + FP))
    recall_main += (TP / (TP + FN))
precision_main /= 3
recall_main /= 3
f1_score_main = 2 * (precision_main * recall_main) / (precision_main + recall_main)

print(f"Accuracy : {accuracy_main:.6f}\nPrecision : {precision_main:.6f}\nRecall : {recall_main:.6f}\nF1-Score : {f1_score_main:.6f}")
