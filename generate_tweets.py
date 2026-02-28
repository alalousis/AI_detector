import time
import ollama
import pandas as pd
from datetime import timedelta

def generate_tweets(desired_model: str, no_posts: int, max_tweet_length: int):
    tweets_df = pd.DataFrame(columns=('id', 'text', 'length'))

    prompt_input = f"Write 1 twitter post about current politics with maximum of {max_tweet_length} characters"    #a
    #prompt_input = f"Write 1 twitter post about politics with maximum of {max_tweet_length} characters" #b
    #cprompt_input = f"Write 1 political tweet with maximum of {max_tweet_length} characters"   #c

    start_time = time.time()
    for i in range(1, no_posts+1):

        # Generate response
        response = ollama.generate(model=desired_model, prompt=prompt_input)

        # Append response to tweets dataset
        tweet = response["response"].strip('"').strip("'") #remove quotes
        tweets_df.loc[i] = [i, tweet, len(tweet)]

    end_time = time.time()
    duration = str(timedelta(seconds=end_time - start_time))

    print(f"Duration:{duration}")

    tweets_df.to_csv(f"datasets/ai_dataset_{no_posts}.csv", index=False, encoding='utf-8')