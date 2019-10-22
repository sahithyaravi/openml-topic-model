import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import FreqDist


import plotly.graph_objs as go
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(["author", "dataset", "feature", "attribute", "source", "cite", "number",
                   "please", "data", "description", "archive",
                   "https", "attributes"])

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
    return col_new


def lemmetize(doc):
    doc = nlp(doc)
    doc_new = []
    for token in doc:
        doc_new.append(token.lemma_)

    return doc_new


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
    fig.show()


# Read df
df = pd.read_pickle('df.pkl')

# Remove stop words
df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")
df["lower"] = lower_case(df["text"])
df['processed'] = remove_stop_words(df['lower'])

pd.set_option('display.max_colwidth', -1)
df["processed"] = df['processed'].apply(lambda x: " ".join([word for word in x.split() if len(word)>2]))
plot_frequency_words(df['processed'])

# Lemmetize reviews
new_text = [lemmetize(doc) for doc in df["processed"]]
df['processed'] = new_text

df.to_pickle("df_proc.pkl")


