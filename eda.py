import pandas as pd
import matplotlib.pyplot as plt 

from tqdm import tqdm
from textblob import TextBlob

import json
import seaborn as sns

class EDA:
    def plot_number_of_words(dataframe:pd.DataFrame, column:str, save_path:str):
        ''' 
        Method to plot a histogram of article/highlight length by splitting based on
        whitespaces. 
        '''
        column_values = dataframe[column]
        
        word_counts = column_values.apply(lambda x:len(x.split()))

        plt.figure(figsize=(10, 6))
        plt.hist(word_counts, bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution of Word Counts in CNN Dataset Summaries')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        
        plt.savefig(f'{save_path}')

    def tag_sentiments(subset_dataframe:pd.DataFrame):
        ''' 
        Tagged for Subset Val and Test

        Annotated using TextBlob. Some annotations are inaccurate, might/may have fixes using human supervision
        '''

        subset_dataframe['sentiment_label'] = ''

        for idx, row in tqdm(subset_dataframe.iterrows()):
            blob = TextBlob(row['article'])
            sentiment = blob.sentiment

            polarity = sentiment.polarity

            if polarity < 0:
                subset_dataframe.at[idx,'sentiment_label'] = 'negative'

            elif polarity > 0:
                subset_dataframe.at[idx,'sentiment_label'] = 'positive'

            else:
                subset_dataframe.at[idx,'sentiment_label'] = 'neutral'

        return subset_dataframe

    def plot_rouge_vs_length_bucket(prediction_json:dict, save_path:str):
                
        df = pd.DataFrame([{
            'ROUGE-L': item['rouge']['rougeL'],
            'Length Bucket': item['length_bucket'],
            'Article Length': len(item['article'])
        } for item in prediction_json])

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Length Bucket', y='ROUGE-L', data=df, ci=None)
        plt.title('ROUGE-L Scores by Length Bucket')
        plt.xlabel('Length Bucket')
        plt.ylabel('ROUGE-L Score')
        plt.savefig(save_path)

    def plot_rouge_vs_category(prediction_json:dict, save_path:str):

        def categorize(bucket):
            if '100' in bucket or '200' in bucket:
                return 'short'
            
            else:
                return 'medium'

        df = pd.DataFrame([{
            'ROUGE-L': item['rouge']['rougeL'],
            'Length Bucket': item['length_bucket'],
            'Article Length': len(item['article'])
        } for item in prediction_json])

        df['article_type'] = df['Length Bucket'].apply(categorize)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='article_type', y='ROUGE-L', data=df, ci=None)
        plt.title('ROUGE-L Scores by Article Type')
        plt.xlabel('Article Type')
        plt.ylabel('ROUGE-L Score')

        plt.savefig(save_path)        

    def plot_rouge_vs_sentiment(prediction_json:dict, save_path:str):

        df = pd.DataFrame([{
            'ROUGE-L': item['rouge']['rougeL'],
            'Sentiment': item['sentiment'],
        } for item in prediction_json])

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Sentiment', y='ROUGE-L', data=df, ci=None)
        plt.title('ROUGE-L Scores by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('ROUGE-L Score')

        plt.savefig(save_path)       

if __name__ == "__main__":

    EDA.plot_rouge_vs_length_bucket(
        json.load(open('final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/subset_val_prediction.json')), 'final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/val_rouge-L_bucket.png'
    )

    EDA.plot_rouge_vs_category(
        json.load(open('final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/subset_val_prediction.json')), 'final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/val_rouge-L_article_type.png'
    )    

    EDA.plot_rouge_vs_sentiment(
        json.load(open('final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/subset_val_prediction.json')), 'final_prediction_samples/26Apr24_Run_Multi_GPU_GPT2-Medium/val_rouge-L_sentiment.png'
    )        