import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import FreqDist
import plotly

import plotly.graph_objs as go
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(["unknown", "target", "uci", "edu", "citation", "html",
                   "policy",
                   "author", "dataset", "feature", "attribute",
                   "source", "cite", "number",
                   "class", "positive", "negative"
                   "please", "data", "description", "archive",
                   "https", "attributes",
                   "prof"])

nlp = spacy.load('en')
# python -m spacy download en using admin on conda prompt


def lower_case(col):
    return ([text.lower() for text in col])


def remove_stop_words(col):
    col_new = []
    for text in col:
        list_of_words = text.split()
        #print(list_of_words)
        new_text = " ".join([i for i in list_of_words if i not in stop_words])
        col_new.append(new_text)
        print(len(text), len(new_text))
    return col_new


def lemmetize(doc):
    doc = nlp(doc)
    doc_new = []
    for token in doc:
        doc_new.append(token.lemma_)

    return " ".join (doc_new)


def plot_frequency_words(col):
    joined_words = " ".join([doc for doc in col])
    all_words = joined_words.split()
    fdist = FreqDist(all_words)
    df = (pd.DataFrame.from_dict(fdist, orient="index"))
    df.reset_index(inplace=True)
    df.columns = ["words", "count"]
    df.sort_values(by="count", inplace=True,ascending=False)
    data = [go.Bar(x=df["words"][:20], y= df["count"][:20])]
    fig = go.Figure(data)
    plotly.offline.plot(fig)


# Read df
df = pd.read_pickle('df.pkl')

# Remove stop words
df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
df["lower"] = lower_case(df["text"])
# Lemmetize reviews
new_text = [lemmetize(doc) for doc in df["lower"]]
df['processed'] = new_text

df['processed'] = remove_stop_words(df['processed'])

pd.set_option('display.max_colwidth', -1)
df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split() if len(word)>2]))
plot_frequency_words(df['processed'])
# Split to list of words
final = []
for doc in df["processed"]:
    final.append(doc.split())

df["processed"] = final
df.to_pickle("df_proc.pkl")

df.to_csv("df_proc.csv")
