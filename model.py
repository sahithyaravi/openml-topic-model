import gensim
from gensim import corpora
import pandas as pd

df = pd.read_pickle("df_proc.pkl")
# Create a term dictionary of our corpus
docs = list(df["processed"].values)
dictionary = corpora.Dictionary(docs)
dictionary.filter_extremes(no_above=0.3)
# Convert to document term matrix
doc_term_mat = [dictionary.doc2bow(doc) for doc in docs]
print(doc_term_mat[0])
print(docs[0])

# LDA
LDA = gensim.models.LdaModel
HDP = gensim.models.HdpModel

lda_model = LDA(corpus=doc_term_mat, passes=50,
                alpha='auto',

                id2word=dictionary, num_topics=4, random_state=1)

hdp_model = HDP(corpus=doc_term_mat,
                id2word=dictionary)
topics = lda_model.print_topics()

print("LDA topics:")
for topic in topics:
    print(topic)

hdp_topics = hdp_model.print_topics()
print("LDA topics:")
for topic in hdp_topics:
    print(topic)