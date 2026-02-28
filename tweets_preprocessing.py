import pandas as pd
import torch
from nltk.tokenize import TweetTokenizer
import re
from torch.utils.data import Dataset


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


def clean_tweets(tweets_df: pd.DataFrame) -> pd.DataFrame:
    tweet_tokenizer = TweetTokenizer()

    if "cleaned_text" not in tweets_df.columns:
        tweets_df["cleaned_text"] = [None] * len(tweets_df)

    for index, row in tweets_df.iterrows():
        tweet_text = row["text"]

        tweet_text = clean_urls(tweet_text)
        tweet_tokenized = tweet_tokenizer.tokenize(tweet_text)
        tweet_tokenized = low_case_words(tweet_tokenized)
        tweet_tokenized = clean_hashtags(tweet_tokenized)
        tweet_tokenized_cleaned = clean_user_mentions(tweet_tokenized)

        tweets_df.at[index, "cleaned_text"] = ' '.join([tweet_tokenized_cleaned][0])

    return tweets_df


class SentimentDataset(Dataset):

    # Class constructor.
    def __init__(self, texts, labels, tokenizer, max_length):
        # self.texts = [clean_text(text) for text in texts]
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Implement the custom len() behaviour for the objects of this class.
    def __len__(self):
        return len(self.texts)

    # Implement the class method that allows slicing within the contents of the
    # custom dataset class.
    def __getitem__(self, idx):
        text = self.texts[idx]
        if len(self.labels) > 0:
            label = self.labels[idx]
        # ---------------------------------------------------------------------
        # Encoding text technical details:
        # ---------------------------------------------------------------------
        # This is a method provided by the BERT tokenizer loaded from the
        # Transformers library. It takes a text string and converts it into
        # input IDs and attention masks necessary for BERT.
        # ---------------------------------------------------------------------
        # Input Parameters:
        # ---------------------------------------------------------------------
        # text: The text to be encoded. This is the individual text string
        #       from the dataset that needs sentiment analysis.
        # add_special_tokens = True: This tells the tokenizer to add special
        #                            tokens like [CLS] at the beginning and
        #                            [SEP] at the end of each text string.
        #                            These tokens are important for BERT as
        #                            [CLS] is often used for classification
        #                            tasks, and [SEP] is a separator token,
        #                            useful especially when dealing with two
        #                            text strings (like in question-answering
        #                            models.
        # ---------------------------------------------------------------------
        # max_length: This sets the maximum length of the tokenized input
        #             sequence. If the text is longer than this, it will be
        #             truncated to max_length. This value is typically set to
        #             the maximum length the model can accept (e.g., 512 tokens
        #             for BERT).
        # ---------------------------------------------------------------------
        # return_token_type_ids = False: This is specific to certain BERT tasks
        #                                that requires differentiating between
        #                                multiple input sequences like questing
        #                                answering tasks. For simple classification
        #                                tasks this value should be set to False.
        # ---------------------------------------------------------------------
        # padding = 'max_length': This ensures that all encoded sequences are
        #                         padded to the same length (max_length). In
        #                         case a sequence is shorter than max_length,
        #                         it will be padded with zeros.
        # ---------------------------------------------------------------------
        # return_attention_mask = True: This directive instructs the tokenizer
        #                               to generate and return attention masks,
        #                               which tell the model which tokens should
        #                               be attended to and which should not
        #                               (e.g. padding tokens)
        # ---------------------------------------------------------------------
        # return_tensors = 'pt': This specifies that the returned tensors should
        #                        PyTorch tensors.
        # ---------------------------------------------------------------------
        # trancation = True: This argument ensures that if a text string is
        #                    longer than the max_length, it will be truncated
        #                    to fit.
        # ---------------------------------------------------------------------
        # Output Parameters:
        # ---------------------------------------------------------------------
        # input_ids: These are the token ids for each token in the text.They
        #            constitute the input for the BERT model.
        # ---------------------------------------------------------------------
        # attention_mask: This is a mask of 1s and 0s indicating which tokens
        #                 are actual words and which are padding.The BERT
        #                 model utilizes this information to know which parts
        #                 of the input it should pay attention to and which
        #                 parts should be ignored.
        # ---------------------------------------------------------------------

        # Encoding text.
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        # Create the dictionary object which will be returned as the item for
        # the requested position.
        # Check whether text sentiment labels are not provided:
        if len(self.labels) == 0:
            item = {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }
        else:
            item = {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        return item