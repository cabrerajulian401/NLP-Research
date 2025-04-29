#!/usr/bin/env python3
# Text analysis with Python
# Using graduate statistics exam answers

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import docx
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def get_exam_directories(base_dir='CAOS Exam'):
    """Get all exam directories from the base directory."""
    try:
        base_path = Path(base_dir)
        if not base_path.exists():
            raise ValueError(f"Base directory not found: {base_dir}")
        
        # Get all directories that start with 'CAOS Exam_'
        exam_dirs = [d for d in base_path.iterdir() 
                    if d.is_dir() and d.name.startswith('CAOS Exam_')]
        
        if not exam_dirs:
            raise ValueError(f"No exam directories found in {base_dir}")
        
        return exam_dirs
    except Exception as e:
        print(f"Error getting exam directories: {str(e)}")
        return []

def read_docx_files(directory):
    """Read all .docx files from a directory and return a list of documents with metadata."""
    documents = []
    for file_path in Path(directory).glob('*.docx'):
        # Extract metadata from filename
        filename = file_path.stem
        parts = filename.split('_')
        metadata = {
            'Problem': parts[0],
            'Type': parts[1] if len(parts) > 1 else None,
            'Image': parts[2] if len(parts) > 2 else None,
            'Version': parts[3] if len(parts) > 3 else None
        }
        
        # Read the document content
        doc = docx.Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        documents.append({
            'text': text,
            'metadata': metadata
        })
    return documents

def preprocess_text(text):
    """Preprocess text by tokenizing, removing stopwords, and converting to lowercase."""
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove punctuation and numbers
    tokens = [token for token in tokens if token.isalpha()]
    
    # Remove single-character tokens
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens

def analyze_documents(documents):
    """Analyze documents and return statistics."""
    stats = []
    for doc in documents:
        text = doc['text']
        tokens = preprocess_text(text)
        sentences = sent_tokenize(text)
        
        stats.append({
            'Version': doc['metadata']['Version'],
            'Types': len(set(tokens)),
            'Tokens': len(tokens),
            'Sentences': len(sentences)
        })
    
    return pd.DataFrame(stats)

def perform_sentiment_analysis(documents):
    """Perform sentiment analysis on documents."""
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    
    for doc in documents:
        sentiment_scores = sia.polarity_scores(doc['text'])
        sentiments.append({
            'Version': doc['metadata']['Version'],
            'Document': doc['metadata']['Problem'],
            'Sentiment': sentiment_scores['compound']
        })
    
    return pd.DataFrame(sentiments)

def analyze_exam_directory(directory):
    """Analyze a single exam directory and return results."""
    try:
        # Get model name from directory name
        model_name = directory.name.split('_')[2] if len(directory.name.split('_')) > 2 else "Unknown"
        
        print(f"\nAnalyzing {model_name} documents...")
        docs = read_docx_files(directory)
        print(f"Found {len(docs)} documents")
        
        # Analyze documents
        stats_df = analyze_documents(docs)
        
        # Perform sentiment analysis
        sentiment_df = perform_sentiment_analysis(docs)
        
        # Create detailed document data
        detailed_df = sentiment_df[['Document', 'Sentiment']].copy()
        detailed_df = detailed_df.rename(columns={'Sentiment': 'Sentiment_Score'})
        
        # Add statistics
        detailed_df['Unique_Words'] = stats_df['Types']
        detailed_df['Total_Words'] = stats_df['Tokens']
        detailed_df['Sentences'] = stats_df['Sentences']
        
        # Sort by document number
        detailed_df['Doc_Num'] = detailed_df['Document'].str.extract('(\d+)').astype(int)
        detailed_df = detailed_df.sort_values('Doc_Num')
        detailed_df = detailed_df.drop('Doc_Num', axis=1)
        
        # Create summary statistics
        summary_stats = {
            'Metric': [
                'Number of Documents',
                'Average Unique Words',
                'Median Unique Words',
                'Min Unique Words',
                'Max Unique Words',
                'Average Total Words',
                'Median Total Words',
                'Min Total Words',
                'Max Total Words',
                'Average Sentences',
                'Median Sentences',
                'Min Sentences',
                'Max Sentences',
                'Average Sentiment',
                'Median Sentiment',
                'Min Sentiment',
                'Max Sentiment',
                'Positive Sentiment Count',
                'Negative Sentiment Count'
            ],
            'Value': [
                len(detailed_df),
                detailed_df['Unique_Words'].mean(),
                detailed_df['Unique_Words'].median(),
                detailed_df['Unique_Words'].min(),
                detailed_df['Unique_Words'].max(),
                detailed_df['Total_Words'].mean(),
                detailed_df['Total_Words'].median(),
                detailed_df['Total_Words'].min(),
                detailed_df['Total_Words'].max(),
                detailed_df['Sentences'].mean(),
                detailed_df['Sentences'].median(),
                detailed_df['Sentences'].min(),
                detailed_df['Sentences'].max(),
                detailed_df['Sentiment_Score'].mean(),
                detailed_df['Sentiment_Score'].median(),
                detailed_df['Sentiment_Score'].min(),
                detailed_df['Sentiment_Score'].max(),
                (detailed_df['Sentiment_Score'] > 0).sum(),
                (detailed_df['Sentiment_Score'] < 0).sum()
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save results
        detailed_df.to_csv(f'CAOS_{model_name}_Detailed.csv', index=False)
        summary_df.to_csv(f'CAOS_{model_name}_Summary_Stats.csv', index=False)
        
        print(f"\n{model_name} Analysis Summary:")
        print("=" * 50)
        for _, row in summary_df.iterrows():
            if isinstance(row['Value'], (int, np.int64)):
                print(f"{row['Metric']}: {row['Value']}")
            else:
                print(f"{row['Metric']}: {row['Value']:.2f}")
        
        return True
    except Exception as e:
        print(f"Error analyzing directory {directory}: {str(e)}")
        return False

def main():
    try:
        # Get all exam directories
        exam_dirs = get_exam_directories()
        
        if not exam_dirs:
            print("No exam directories found to analyze.")
            return
        
        print(f"Found {len(exam_dirs)} exam directories to analyze.")
        
        # Analyze each directory
        successful_analyses = 0
        for directory in exam_dirs:
            if analyze_exam_directory(directory):
                successful_analyses += 1
        
        print(f"\nAnalysis complete! Successfully analyzed {successful_analyses} out of {len(exam_dirs)} directories.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()