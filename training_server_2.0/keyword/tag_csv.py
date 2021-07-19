from eunjeon import Mecab
import pandas as pd

dataset = pd.read_csv("../csv/AM_210329_COVID2.csv", encoding="utf-8")

message_list = dataset["MESSAGE"].values.tolist()
category_list = dataset["CATEGORY"].values.tolist()
sub_list = dataset["SUB"].values.tolist()

# Mecab
mecab = Mecab()

noun_message_list = []
for message in message_list:
    message_split = message.split("]")
    new_message = message_split[1]
    if len(message_split) > 2:
        for message_piece in message_split[2:]:
            new_message = f"{new_message}]{message_piece}"
    # noun_list = mecab.nouns(message.split("]")[1])
    noun_list = mecab.nouns(new_message)
    if len(noun_list) > 0:
        message_noun = noun_list[0]
        for noun in noun_list[1:]:
            message_noun = f"{message_noun} {noun}"
    else:
        message_noun = " "
    noun_message_list.append(message_noun)
    # print(message_noun)

# new_dataset = [category_list, sub_list, message_list]
new_dataset = {"CATEGORY": category_list, "SUB": sub_list, "MESSAGE": noun_message_list}
new_df = pd.DataFrame(new_dataset)

new_df.to_csv("./COVID-nouns.csv", encoding="utf-8")
