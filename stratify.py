import pandas as pd
from tqdm import tqdm

def truncate_dataframe(dataframe:pd.DataFrame, truncate_length:int):
    truncated_dataframe = pd.DataFrame(columns=dataframe.columns)

    for idx, row in dataframe.iterrows():
        article_word_count = len(row['article'].split())
        if article_word_count < truncate_length:
            truncated_dataframe = truncated_dataframe._append(
                row, ignore_index=True
            )

    return truncated_dataframe

def categorize(word_count):
    if word_count <= 300:
        return 'short'
    elif 400 <= word_count <= 600:
        return 'medium'
    else:  # Covers cases where word count > 600
        return 'long'    

def create_article_length_column(dataframe:pd.DataFrame):
    if 'article_length' not in dataframe.columns:
        dataframe['article_length'] = dataframe['article'].apply(
            lambda x : len(x.split())
        )

    return dataframe

def create_subset(dataframe:pd.DataFrame, n_percent:float, bucket:int=100):
    
    subset_dataframe = pd.DataFrame(columns=dataframe.columns)

    if 'article_length' not in dataframe.columns:
        dataframe['article_length'] = 0        
        dataframe['article_length'] = dataframe['article'].apply(lambda x: len(x.split()))

    else:
        bucket_intervals = range(0, dataframe['article_length'].max() + bucket, bucket)
        dataframe['length_bucket'] = pd.cut(
            dataframe['article_length'], bins=bucket_intervals, right=False
        )

        subset_dataframe = dataframe.groupby('length_bucket').apply(
            lambda x : x.sample(max(int(n_percent * len(x)), 1), replace=False).reset_index(drop=True)
        )

    subset_dataframe['article_length_category'] = subset_dataframe['article_length'].apply(
        lambda x : categorize(x)
    )        

    return subset_dataframe
        

