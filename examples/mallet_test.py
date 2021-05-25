from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet, ldamallet
from gensim import corpora
import pandas as pd
from gensim.models import CoherenceModel

import pyLDAvis.gensim
import pyLDAvis
from src.utils import plot_word_cloud
df = pd.read_pickle("data/df_proc.pkl")
docs = list(df["processed"].values)

docs_train = docs[:2000]
docs_test = docs[2000:]
dictionary = corpora.Dictionary(docs_train)

# Filter terms that occur in more than 50% of docs
dictionary.filter_extremes(no_above=0.5)

# Convert to document term matrix (corpus)
doc_term_mat_train = [dictionary.doc2bow(doc) for doc in docs_train]
doc_term_mat_test = [dictionary.doc2bow(doc) for doc in docs_test]

path_to_mallet_binary = r'C:\mallet\bin\mallet'
if __name__ == "__main__":
    model = LdaMallet(path_to_mallet_binary, corpus=doc_term_mat_train, alpha=5,
                      num_topics=10, id2word=dictionary, optimize_interval=50)

    topics = model.print_topics()
    for topic in topics:
        print(topic)

    # Compute Coherence Score for base model
    coherence_model_lda = CoherenceModel(model=model,
                                         corpus=doc_term_mat_train,
                                         texts=docs_train,
                                         dictionary=dictionary,
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    gensim_model = ldamallet.malletmodel2ldamodel(model)
    # Visualize the topics
    vis_prepared = pyLDAvis.gensim.prepare(gensim_model, doc_term_mat_train, dictionary)
    pyLDAvis.save_html(vis_prepared, "mallet.html")
    print('\nCoherence Score: ', coherence_lda)
    plot_word_cloud(gensim_model)


