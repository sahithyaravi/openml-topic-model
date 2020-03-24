import re
import gensim


def remove_author_info(df):
    """

    :param df: input data
    :return: Remove author related information
    """
    out = []
    for text in df['text']:
        split = text.splitlines()
        if len(split) > 3 and "author" in split[0]:
            out_text = " ".join(split[3:])
        else:
            out_text = text
        out.append(out_text)
    df["text"] = out

    return df


def lower_case(df):
    df["text"] = [text.lower() for text in df["text"]]
    return df


def remove_special_chars(df):
    """

    :param df: input df with 'text' column
    :return: df with special chars removed from text column
    """
    # Remove url
    df['text'] = [re.sub(r"http\S+", "", text) for text in df["text"]]

    # Remove special chars and numbers
    df['text'] = df['text'].str.replace("[^a-zA-Z#]", " ")

    # Remove emails:
    df["text"] = [re.sub('\S*@\S*\s?', '', text) for text in df["text"]]
    df["len"] = df["text"].str.len()
    return df


def remove_stop_words(col, stop_words):
    """

    :param col: The pd.series containing text
    :param stop_words: the stopwords that need to be removed
    :return:
    """
    col_new = []
    for text in col:
        list_of_words = text.split()
        new_text = " ".join([i for i in list_of_words if i not in stop_words])
        col_new.append(new_text)
    return col_new


def get_bigrams(final):
    """

    :param final: Input text data
    :return: Text with bigrams
    """
    # Bigrams
    bigram = gensim.models.Phrases(final, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    final = [bigram_mod[line] for line in final]
    return final
