import pandas as pd
from nltk.tokenize import TweetTokenizer
import re


def clean_user_mentions(tweet_tokenized: list[str]) -> list[str]:
    tweet_tokenized_cleaned = ["__user_mention__" if t.startswith("@") and len(t) > 1 else t for t in tweet_tokenized]
    tweet_tokenized_cleaned = ["__invalid_user_mention__" if t.startswith("@") and len(t) <= 1 else t for t in tweet_tokenized_cleaned ]
    return tweet_tokenized_cleaned


def clean_urls(tweet_text: str) -> str:
    tweet_text = re.sub(r'http\S+', '__url__', tweet_text)
    return tweet_text


def low_case_words(tweet_tokenized: list[str]) -> list[str]:
    tweet_tokenized_lowered = [t.lower() for t in tweet_tokenized]
    return tweet_tokenized_lowered
    
    
def clean_hashtags(tweet_tokenized: list[str]) -> list[str]:
    tweet_tokenized_cleaned = [t for t in tweet_tokenized if not t.startswith("#")]
    return tweet_tokenized_cleaned


def tokenize_tweets_dataset(tweets_df: pd.DataFrame) -> pd.DataFrame:
    tweet_tokenizer = TweetTokenizer()

    if "text_tokenized" not in tweets_df.columns:
        tweets_df["text_tokenized"] = [None] * len(tweets_df)

    for index, row in tweets_df.iterrows():
        tweet_text = row["text"]
        
        tweet_text = clean_urls(tweet_text)
        tweet_tokenized = tweet_tokenizer.tokenize(tweet_text)
        tweet_tokenized = low_case_words(tweet_tokenized)
        tweet_tokenized = clean_hashtags(tweet_tokenized)
        tweet_tokenized_cleaned = clean_user_mentions(tweet_tokenized)
        
        tweets_df.at[index, "text_tokenized"] = [tweet_tokenized_cleaned][0]

    return tweets_df