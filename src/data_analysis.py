import pandas as pd
import numpy as np
from datasets import load_dataset
import random


def perform_full_eda():
    print("Downloading dataset: clarin-pl/twitteremo...")
    
    raw_dataset = load_dataset("clarin-pl/twitteremo")
    
    print('=' * 50)
    print("\nDataset Structure:\n")
    print('=' * 50)

    print(raw_dataset)
    
    df = pd.DataFrame(raw_dataset['train'])
    
    text_col = 'tekst'
    
    print('=' * 50)
    print('Example text:')
    print('=' * 50)

    for _ in range(3):
        j = random.randint(0, len(df))
        print(f"Tweet {j}: {df[text_col].iloc[j]}")
        print("-" * 30)

    print('=' * 50)
    print('Tweets text length analysis')
    print('=' * 50)


 
    word_counts = df[text_col].apply(lambda x: len(str(x).split()))
    print(f"Average words count in a tweet: {word_counts.mean():.1f}")
    print(f"Median words count: {word_counts.median():.0f}")
    print(f"Max words count: {word_counts.max()}")

    print('=' * 50)
    print('Emotions label analysis')
    print('=' * 50)

    ignored_cols = [text_col, 'data', 'id']
    label_cols = [col for col in df.columns if col not in ignored_cols]
    
    print(f"Emotion cols: ({len(label_cols)} klas): {label_cols}")
    
    for col in label_cols:
        df[col] = df[col].astype(float)
    

    emotion_sums = df[label_cols].sum().sort_values(ascending=False)
    emotion_percentages = (df[label_cols].sum() / len(df) * 100).sort_values(ascending=False)
    
    print("\nNumber of occurences of individual emotions:")
    for emotion in emotion_sums.index:
        print(f" - {emotion}: {int(emotion_sums[emotion])}  ({emotion_percentages[emotion]:.1f}% of all tweets)")

    emotions_per_tweet = df[label_cols].sum(axis=1)
    print(f"\nAverage number of emotions in one tweet: {emotions_per_tweet.mean():.2f}")
    print(f"Max number of emotions in one tweet: {emotions_per_tweet.max()}")
    print(f"Number of tweets without any emotion: {sum(emotions_per_tweet == 0)}")

if __name__ == "__main__":
    perform_full_eda()