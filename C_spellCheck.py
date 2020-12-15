import pandas as pd
from hanspell import spell_checker
from tqdm import tqdm

train_data = pd.read_csv('Vec.csv', encoding='utf-8').iloc[:]
train_data.index = range(len(train_data))

train_data_text = list(train_data['내용'])

train_clear_text = []

for i in tqdm(range(len(train_data_text))):
    try:
        result_train = spell_checker.check(train_data['내용'][i])
        train_data['내용'][i] = result_train.as_dict()['checked']
    except:
        pass

print(train_data.head())

train_data.to_csv('data.csv', mode='w', encoding='utf-8')
