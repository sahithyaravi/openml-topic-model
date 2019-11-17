import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import FreqDist
import plotly
import re
import gensim
import plotly.graph_objs as go
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(["example", "experiment", "sample", "problem", "input", "output", "set", "task", "study",
                   "training", "prediction", "model", "test", "train",
                   "author", "source", "https", "uci", "edu", "citation", "html", "policy", "datum", "please",
                   "title", "dataset", "feature", "attribute", "attributes", "row", "column", "image", "file", "pixel",
                   "description", "cite", "publication", "result", "distribution", "point",
                   "nominal", "enum", "string", "categorical", "number", "continuous", "numeric", "variable",
                   "instance", "set", "classtype", "none", "note", "inf", "information", "type", "data",
                   "target", "class", "positive", "negative", "value",
                   "time", "date", "year",
                   "imputation", "classification", "regression",
                   "colinearity", "degree", "average",
                   "unknown", "several", "version", "original",
                   "name", "project", "program", "paper", "thesis", "database", "format"
                   ])

nlp = spacy.load('en')
# python -m spacy download en using admin on conda prompt


def remove_url(col):
    col_url = [re.sub(r"http\S+","", text) for text in col]
    return col_url    #\S+ matches all whitespace characters


def lower_case(col):
    return [text.lower() for text in col]


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
    sents = nlp(doc)
    doc_new = []
    for token in sents:
        if token.pos_ in ['NOUN']:
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
df = pd.read_pickle('df_unique.pkl')
# Lower case
df["text"] = lower_case(df["text"])

# Remove author line
out =[]
for text in df['text']:
    split = text.splitlines()
    if len(split) > 3 and "author" in split[0]:
        out_text = " ".join(split[3:])
    else:
        out_text = text
    out.append(out_text)
df["text"] = out

# Remove url
df['text'] = remove_url(df['text'])

# Remove special chars and numbers
df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")

# Remove emails:
df["text"] = [re.sub('\S*@\S*\s?', '', text) for text in df["text"]]
df["text"] = df["text"] + df["name"]

# Lemmetize
df['processed'] = [lemmetize(doc) for doc in df["text"]]


# Remover short words
df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 2]))

df['processed'] = remove_stop_words(df['processed'])

plot_frequency_words(df['processed'])
# Split to list of words
final = []
for doc in df["processed"]:
    final.append(doc.split())

# Bigrams
bigram = gensim.models.Phrases(final, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
final = [bigram_mod[line] for line in final]

df["processed"] = final
df["title1"] = df["name"].str.split()
df["processed"] = df["title1"] + df["processed"]

df.to_pickle("df_proc.pkl")

df.to_csv("df_proc.csv")
