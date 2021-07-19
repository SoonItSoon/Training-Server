from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

USE_SAVED_MODEL = False

# dataset = pd.read_csv("csv/AM_210329_COVID7.csv", encoding="utf-8")
# data = dataset["MESSAGE"].values.tolist()
data = fetch_20newsgroups(subset='all')['data']

if USE_SAVED_MODEL:
    model = BERTopic.load("bertopic_model.pt")
else:
    # model = BERTopic(language="Korean")
    model = BERTopic(language="multilingual", embedding_model="distiluse-base-multilingual-cased-v1", nr_topics=5)
    # model = BERTopic(nr_topics=5)
    topics, probs = model.fit_transform(data)
    model.save("bertopic_model.pt")

topic_freq = model.get_topic_freq()
print(len(topic_freq), topic_freq)
