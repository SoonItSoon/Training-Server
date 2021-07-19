import pandas as pd

dataset = pd.read_csv("../csv/AM_210329_COVID5.csv", encoding="utf-8")

message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
sub_list = dataset["SUB"].values.tolist()

MAX_LEN = 10
PERCENTAGE = 0.4

keyword_1 = ["확진", "발생", "동선", "역학", "접촉", "코로나", "공개", "홈페이지", "방역", "예정"]
keyword_2 = ["보건소", "방문", "검사", "선별", "진료소", "확진", "이용", "상담", "코로나", "교회"]
keyword_3 = ["착용", "자제", "코로나", "마스크", "방문", "수칙", "방역", "모임", "예방", "지역"]
keyword_4 = ["자제", "코로나", "지역", "방문", "착용", "확진", "발생", "모임", "감염", "마스크"]
keyword_5 = ["운영", "착용", "시설", "금지", "단계", "선별", "코로나", "사회", "방역", "두기"]


def getMainCategory(count, message):
    count1 = count["1"] / MAX_LEN
    count2 = count["2"] / MAX_LEN
    count3 = count["3"] / MAX_LEN
    count4 = count["4"] / MAX_LEN
    count5 = count["5"] / MAX_LEN
    main_category = []
    if count1 >= PERCENTAGE:
        main_category.append(1)
    if count2 >= PERCENTAGE:
        main_category.append(2)
    if count3 >= PERCENTAGE:
        main_category.append(3)
    if count4 >= PERCENTAGE:
        main_category.append(4)
    if count5 >= PERCENTAGE:
        main_category.append(5)
    if len(main_category) == 1:
        return main_category[0]
    elif len(main_category) == 0:
        return 6
    else:
        for i in range(0, MAX_LEN):
            for j in main_category:
                if j == 1 and keyword_1[i] in message:
                    return 1
                elif j == 2 and keyword_2[i] in message:
                    return 2
                elif j == 3 and keyword_3[i] in message:
                    return 3
                elif j == 4 and keyword_4[i] in message:
                    return 4
                elif j == 5 and keyword_5[i] in message:
                    return 5
    return 6


main_dict = {"확진자발생": 1, "보건소방문": 2, "개인방역수칙": 3, "발생현황": 4, "행정안내": 5}

con_matrix_main = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
accuracy_main = 0
for i in range(0, len(message_list)):
    message = message_list[i]
    count = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    for j in range(0, MAX_LEN):
        if keyword_1[j] in message:
            count["1"] += 1
        if keyword_2[j] in message:
            count["2"] += 1
        if keyword_3[j] in message:
            count["3"] += 1
        if keyword_4[j] in message:
            count["4"] += 1
        if keyword_5[j] in message:
            count["5"] += 1
    main_category = getMainCategory(count, message)
    if category_list[i] == "안내":
        category_list[i] = sub_list[i]
    con_matrix_main[main_dict[category_list[i]] - 1][main_category - 1] += 1
    if main_dict[category_list[i]] == main_category:
        accuracy_main += 1

print(accuracy_main, len(message_list))
accuracy_main /= len(message_list)
precision_main = 0
recall_main = 0
f1_score_main = 0
print(accuracy_main)
for i in range(0, 5):
    print(con_matrix_main[i])
    TP = con_matrix_main[i][i]
    FP = 0
    FN = 0
    for j in range(0, 6):
        if j != i:
            FP += con_matrix_main[j][i]
            FN += con_matrix_main[i][j]
    precision_main += (TP / (TP + FP))
    recall_main += (TP / (TP + FN))
precision_main /= 5
recall_main /= 5
f1_score_main = 2 * (precision_main * recall_main) / (precision_main + recall_main)

print(f"Accuracy : {accuracy_main:.6f}\nPrecision : {precision_main:.6f}\nRecall : {recall_main:.6f}\nF1-Score : {f1_score_main:.6f}")
