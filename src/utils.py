import gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import CoherenceModel
from gensim.test.utils import datapath


def compute_coherence_score(corpus, id2word, num_topics, alpha, eta, text, test):
    if alpha == 'auto' or eta == 'auto':
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                           iterations=1000,
                                           id2word=id2word,
                                           alpha=alpha,
                                           eta=eta,
                                           num_topics=num_topics,
                                           random_state=100,
                                           passes=200,
                                           )
    else:
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               iterations=1000,
                                               workers=4,
                                               id2word=id2word,
                                               alpha=alpha,
                                               eta=eta,
                                               num_topics=num_topics,
                                               random_state=100,
                                               passes=200,
                                               )

    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=corpus,
                                         dictionary=id2word,
                                         texts=text)
    perplexity = lda_model.log_perplexity(test)
    return coherence_model_lda.get_coherence(), perplexity, lda_model


def plot_word_cloud(lda, folder_path = ""):
    # tempfile = datapath("model")

    wc = WordCloud(
        background_color="white",
        # max_words=2000,
        # width=1024,
        # height=720,
        # stopwords=stop_words
    )

    # lda = gensim.models.LdaMulticore.load(tempfile)

    fig, axes = plt.subplots(2, round(lda.num_topics/2), figsize=(20, 20), sharex=True, sharey=True)
    fig.delaxes(axes[1, 3])
    words = {}
    for t, ax in enumerate(axes.flatten()):
        if t != lda.num_topics:
            # plt.figure()
            fig.add_subplot(ax)
            top_words = dict(lda.show_topic(t, 20))
            words.update(top_words)
            plt.gca().imshow(wc.fit_words(top_words))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.savefig(folder_path + 'wc_topics.png')
    plt.show()
    plt.subplots(figsize=(8, 8))
    plt.imshow(wc.fit_words(words))
    plt.savefig(folder_path + 'WC.png')
    plt.show()
