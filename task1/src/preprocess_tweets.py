from emoji import demojize
from nltk.tokenize import TweetTokenizer
from functools import partial
from tqdm import tqdm

tqdm.pandas()

tweettokenizer = TweetTokenizer()

def normalize_tweet(tweet):
    norm_tweet = (
        tweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    norm_tweet = (
        norm_tweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    norm_tweet = (
        norm_tweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return norm_tweet


def replace_user_handles(tweet, replace='@USER'):
    tokens = tweettokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if token.startswith("@"):
            new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def replace_urls(tweet, replace='HTTPURL'):
    tokens = tweettokenizer.tokenize(tweet)

    if type(replace) == str:
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                new_tokens.append(replace)
            else:
                new_tokens.append(token)

    elif type(replace) == list:
        n_replaced_tokens = 0
        new_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.startswith("http") or lower_token.startswith("www"):
                if n_replaced_tokens < len(replace):
                    new_tokens.append(replace[n_replaced_tokens])
                    n_replaced_tokens += 1
                else:
                    new_tokens.append(token)
            else:
                new_tokens.append(token)

        while n_replaced_tokens < len(replace):
            new_tokens.append(replace[n_replaced_tokens])
            n_replaced_tokens = n_replaced_tokens + 1

    return " ".join(new_tokens)


def replace_emojis(tweet, replace='demojize'):
    tokens = tweettokenizer.tokenize(tweet)

    new_tokens = []
    for token in tokens:
        if len(token) == 1:
            if replace == 'demojize':
                new_tokens.append(demojize(token))
            else:
                new_tokens.append(replace)
        else:
            new_tokens.append(token)

    return " ".join(new_tokens)


def preprocess_tweets(data, lowercase=True, normalize=True, emojis='demojize', user_handles='@USER', urls='original_urls'):
    if lowercase:
        data["text"] = data["text"].str.lower()

    if normalize:
        data["text"] = data["text"].progress_apply(normalize_tweet)

    if emojis:
        data["text"] = data["text"].progress_apply(partial(replace_emojis, replace=emojis))

    if user_handles:
        data["text"] = data["text"].progress_apply(partial(replace_user_handles, replace=user_handles))

    if urls == 'original_urls':
        data["text"] = data[["text", 'processed_urls']].progress_apply(lambda row: replace_urls(*row), axis=1)

    elif urls:
        data["text"] = data["text"].progress_apply(partial(replace_urls, replace=urls))

    return data