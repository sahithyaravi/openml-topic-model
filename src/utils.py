import gensim
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models import CoherenceModel
from gensim.test.utils import datapath


def compute_coherence_score(corpus, id2word, num_topics, alpha, eta, text, test):
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           workers=3,
                                           id2word=id2word,
                                           alpha=alpha,
                                           eta=eta,
                                           num_topics=num_topics,
                                           random_state=100,
                                           chunksize=100,
                                           passes=200,
                                           per_word_topics=True)
    coherence_model_lda = CoherenceModel(model=lda_model,
                                         coherence='c_v',
                                         corpus=corpus,
                                         dictionary=id2word,
                                         texts=text)
    perplexity = lda_model.log_perplexity(test)
    return coherence_model_lda.get_coherence(), perplexity


def plot_word_cloud():
    tempfile = datapath("model")

    wc = WordCloud(
        # background_color="white",
        # max_words=2000,
        # width=1024,
        # height=720,
        # stopwords=stop_words
    )

    lda = gensim.models.LdaMulticore.load(tempfile)

    fig, axes = plt.subplots(2, round(lda.num_topics/2), figsize=(20, 20), sharex=True, sharey=True)
    fig.delaxes(axes[1, 3])
    words = {}
    for t, ax in enumerate(axes.flatten()):
        if t != lda.num_topics:
            # plt.figure()
            fig.add_subplot(ax)
            top_words = dict(lda.show_topic(t, 200))
            words.update(top_words)
            print("top_words\n", top_words)
            plt.gca().imshow(wc.fit_words(top_words))
            # plt.gca().axis("off")
            # plt.gca().set_title("Topic #"+str(t))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    plt.subplots(figsize =(8, 8))
    plt.imshow(wc.fit_words(words))
    plt.savefig('WC.png')
    plt.show()
