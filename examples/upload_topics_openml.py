import pandas as pd
pd.set_option('display.max_colwidth', 1000)
from openml.datasets.functions import _topic_add_dataset, _topic_delete_dataset
# Choose result folder containing the topic assignments
RESULT_FOLDER = '../results/'
RESULT_PATH = 'NOUNVERB_filter_0.6_good/resultdf.csv'
df = pd.read_csv(RESULT_FOLDER + RESULT_PATH)

# Look at different topics
number_of_topics = df['best_topic'].nunique()
print("Number of topics", number_of_topics)
topicid = 0
print("Keywords of topic", df[df['best_topic'] == topicid]["topic_info"].head(10))


# keys = list(range(-1, number_of_topics))
# you have to create a manual mapping of topic numbers to topics
# Manually examine keywords and documents under each topic to determine the topic name
# values = ['challenge']
# map_dict = dict(zip(keys, values))
# df = df[df["best_topic"] != -1]

import openml
openml.config.apikey = "5f0b74b33503e4ad4a7181a91e28719f"


def add_topic(topicname, topicid):
    df_topic = df[df["best_topic"] == topicid]
    for index, row in df_topic.iterrows():
      print(_topic_add_dataset(row['id'], topicname))

def delete_topic(topicname, topicid):
    df_topic = df[df["best_topic"] == topicid]
    for index, row in df_topic.iterrows():
      print(_topic_delete_dataset(row['id'], topicname))

# add_topic('Artificial datasets', 14)
# add_topic('Drug discovery', 10)
# add_topic('Gene expression', 8)
# add_topic('Trading', 7)
# add_topic('Health', 6)
# add_topic('Book-based', 3)

