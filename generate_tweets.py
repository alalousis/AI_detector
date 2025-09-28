import time
import ollama
import pandas as pd
from datetime import timedelta



def generate_tweets(desired_model: str, no_posts: int, max_tweet_length: int):
    tweets_df = pd.DataFrame(columns=('id', 'text', 'length'))

    # Define model and initialize chat history
    # desired_model = "llama3.2"
    POSTS_NUMBER = 1000
    MAX_TWEET_LENGTH = 280

    prompt_input = f"Write 1 political twitter post with maximum of {max_tweet_length} characters"

    start_time = time.time()
    for i in range(1, no_posts+1):

        # Generate response
        response = ollama.generate(model=desired_model, prompt=prompt_input)

        # Append response to tweets dataset
        tweet = response["response"]
        tweets_df.loc[i] = [i, tweet, len(tweet)]

    end_time = time.time()
    duration = str(timedelta(seconds=end_time - start_time))

    print(f"Duration:{duration}")