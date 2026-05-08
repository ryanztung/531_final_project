import pandas as pd
import os

BASE_DIRECTORY = './Resumes/Output_Resumes'

batch_summaries = []

# Walk through each subdirectory
for root, dirs, files in os.walk(BASE_DIRECTORY):
    if 'summary.csv' in files:
        # Retrieve summary path
        summary_path = os.path.join(root, 'summary.csv')
        df = pd.read_csv(summary_path)

        # Store dataframe
        batch_summaries.append(df)

# Combine summary data
df_summaries = pd.concat(batch_summaries, ignore_index=True)

# Load annotated dataset
df_corpus = pd.read_excel('./Resumes/CareerCorpus.xlsx')
df_corpus.rename(columns={'ID': 'resume_id'}, inplace=True)
df_corpus = df_corpus.drop_duplicates(subset='resume_id')

# Compute mean annotation scores
df_corpus['human_score'] = df_corpus[['Annotator-1', 'Annotator-2']].mean(axis=1).round(2)

# Join dataframes
df = pd.merge(df_summaries, df_corpus, on='resume_id')
df = df[['id', 'resume_id', 'name', 'race', 'gender', 'human_score']]

# Save full summary to CSV
df.to_csv(f'{BASE_DIRECTORY}/output_summary.csv', index=False)