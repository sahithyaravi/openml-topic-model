import gensim
from gensim import corpora
import pandas as pd

df = pd.read_pickle("df_proc.pkl")
# Create a term dictionary of our corpus
docs = list(df["processed"].values)
dictionary = corpora.Dictionary(docs)
# Convert to document term matrix
doc_term_mat = [dictionary.doc2bow(doc) for doc in docs]
print(doc_term_mat[0])
print(docs[0])

# LDA
LDA = gensim.models.ldamodel.LdaModel

lda_model = LDA(corpus=doc_term_mat, id2word=dictionary, num_topics=10, random_state=1)
print(lda_model.print_topics())