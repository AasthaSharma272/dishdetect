"""
Food Classification Prediction
--------------------------------------
Author: Aastha Sharma, Narges Movahedian Nezhad, Raghav Singhal, Jessica Song Li
CSC311 Machine Learning Project

Description:
Hybrid text+structured model for food item classification (Pizza/Shawarma/Sushi).
Combines:
- Naive Bayes on text features (movie/drink columns)
- Logistic Regression on structured features (Q1-Q4, Q7-Q8)
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path
import re


# ---Data cleaning globals ---
word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
    'eighteen': 18, 'nineteen': 19, 'twenty': 20
}

word_to_num_Q4 = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

all_q3_categories = ['Week day dinner', 'Weekend lunch', 'Late night snack', 'Week day lunch', 'Weekend dinner', 'At a party']

all_q7_categories = ['Teachers', 'Friends', 'Parents', 'Siblings', 'Strangers']

# --- Data Cleaning Functions ---
def extract_number(text):
    """Used to clean Q2 to extract number from text"""
    if pd.isna(text):
        return np.nan

    text = str(text).lower().strip()

    for word, num in word_to_num.items():
        text = re.sub(rf"\b{word}\b", str(num), text)

    numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]

    if len(numbers) == 0:
        return np.nan
    else:
        return max(numbers)

def count_ingredients_from_text(text):
    """Used to clean Q2 after extract number to get ingredients from text"""
    if pd.isna(text):
        return np.nan

    text = str(text).lower().strip()

    text = re.sub(r'[\n\*\•]', ',', text)
    text = re.sub(r'\s+and\s+', ',', text)
    text = re.sub(r'\s+', ' ', text)

    parts = [p.strip() for p in text.split(',')]
    ingredients = [p for p in parts if len(p) > 1 and not p.isspace()]

    count = len(ingredients)
    if ':' in text:
        count -= 1

    return count if count > 0 else np.nan

def map_q8_hot_sauce(text):
    """
    Maps Q8 hot sauce preferences to numeric values.
    """
    q8_mapping = {
        'A little (mild)': 2,
        'A moderate amount (medium)': 3,
        'A lot (hot)': 4,
        'I will have some of this food item with my hot sauce': 5
    }

    return text.map(q8_mapping).fillna(1)

def extract_price(text):
    """Function used in Q4 to extract price from text"""
    if pd.isna(text):
        return np.nan

    text = str(text).lower()
    text = re.sub(r'[^\w\s\.\$\-–—]', '', text)
    text = text.strip()

    for word, num in word_to_num_Q4.items():
        text = re.sub(rf'\b{word}\b', str(num), text)

    text = re.sub(r'cad|canadian dollars?|bucks|dollars?|usd|\$', '', text)

    text = re.sub(r'(\d+(?:\.\d+)?)[\s]*[-–—to]+[\s]*(\d+(?:\.\d+)?)', r'[\1,\2]', text)

    matches = re.findall(r'\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]', text)
    if matches:
        nums = [float(n) for n in matches[0]]
        return np.mean(nums)

    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        return float(numbers[0])

    return np.nan

def preprocess_data(df):
    """
    Cleans and prepares the raw input data matching the training format
    """
    # Column Renaming
    column_mapping = {
        'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': 'Q1',
        'Q2: How many ingredients would you expect this food item to contain?': 'Q2',
        'Q4: How much would you expect to pay for one serving of this food item?': 'Q4',
        'Q8: How much hot sauce would you add to this food item?': 'Q8',
        'Q5: What movie do you think of when thinking of this food item?': 'Q5',
        'Q6: What drink would you pair with this food item?': 'Q6',
        'Q3: In what setting would you expect this food to be served? Please check all that apply': 'Q3',
        'Q7: When you think about this food item, who does it remind you of?': 'Q7'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Q1 cleaning
    df['Q1'] = df['Q1'].fillna(1)

    # Q2 cleaning
    df['Q2_clean'] = df['Q2'].apply(extract_number)
    df.loc[df["Q2_clean"].isna(), "Q2_clean"] = df.loc[df["Q2_clean"].isna(), "Q2"].apply(count_ingredients_from_text)
    df.drop('Q2', axis=1, inplace=True)
    df.rename(columns={'Q2_clean': 'Q2'}, inplace=True)
    median_price = df['Q2'].median()
    df['Q2'] = df['Q2'].fillna(median_price)

    # Q3 cleaning
    for category in all_q3_categories:
        df[f'Q3_{category}'] = df['Q3'].apply(
            lambda x: 1 if isinstance(x, str) and category in x else 0
        )
    df.drop('Q3', axis=1, inplace=True)

    # Q4 cleaning
    df["Q4_pay_clean"] = df["Q4"].apply(extract_price)
    df.drop('Q4', axis=1, inplace=True)
    df.rename(columns={'Q4_pay_clean': 'Q4'}, inplace=True)
    median_price = df['Q4'].median() 
    df['Q4'] = df['Q4'].fillna(median_price)

    # Q5 cleaning
    df["Q5"] = df["Q5"].fillna('')

    # Q6 cleaning
    df["Q6"] = df["Q6"].fillna('')

    # Q7 cleaning
    for category in all_q7_categories:
        df[f'Q7_{category}'] = df['Q7'].apply(
            lambda x: 1 if isinstance(x, str) and category in x else 0
        )
    df.drop('Q7', axis=1, inplace=True)

    # Q8 cleaning
    df['Q8_hot_sauce_numeric'] = map_q8_hot_sauce(df['Q8'])
    df.drop('Q8', axis=1, inplace=True)
    df.rename(columns={'Q8_hot_sauce_numeric': 'Q8'}, inplace=True)

    return df

# --- Model Loading (One-time setup) ---
model_dir = Path('model_assets')

# 1. Load Logistic Regression Components
lr_weights = np.load(model_dir/'lr_weights.npy')
lr_intercept = np.load(model_dir/'lr_intercept.npy')
feature_means = np.load(model_dir/'feature_means.npy')
feature_stds = np.load(model_dir/'feature_stds.npy')

# 2. Load Naive Bayes Components
with open(model_dir/'nb_word_counts.json') as f:
    nb_word_counts = json.load(f)
nb_class_priors = np.load(model_dir/'nb_class_priors.npy')

# 3. Load TF-IDF Components
with open(model_dir/'tfidf_vocab.json') as f:
    tfidf_vocab = json.load(f)
tfidf_idf = np.load(model_dir/'tfidf_idf.npy')

# 4. Load Label Mapping and Categories
with open(model_dir/'label_mapping.json') as f:
    label_mapping = json.load(f)

# --- Core Prediction Functions ---
def preprocess_text(text):
    """Tokenization matching your training pipeline"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()  # Simple whitespace tokenizer

def compute_tfidf(tokens):
    """Manual TF-IDF vectorization"""
    word_counts = {}
    for word in tokens:
        if word in tfidf_vocab:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    vector = np.zeros(len(tfidf_vocab))
    for word, count in word_counts.items():
        vector[tfidf_vocab[word]] = count * tfidf_idf[tfidf_vocab[word]]
    return vector

def predict_nb(text):
    """Naive Bayes prediction from word counts"""
    tokens = preprocess_text(text)
    tfidf_vec = compute_tfidf(tokens)
    
    log_probs = nb_class_priors.copy()
    for cls in label_mapping:
        cls_idx = int(cls)
        for word, vec_idx in tfidf_vocab.items():
            if tfidf_vec[vec_idx] > 0:
                log_probs[cls_idx] += nb_word_counts[cls].get(word, -20)
    
    return np.exp(log_probs - np.max(log_probs))

def predict_lr(features):
    """Logistic regression prediction"""
    scores = np.dot(features, lr_weights.T) + lr_intercept
    return np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))

def predict_single(row):
    """Predict one row of data"""
    # Text prediction
    text = f"{row['Q5']} {row['Q6']}"
    nb_probs = predict_nb(text)
    
    # LR prediction
    # Scale only the numeric features
    numeric_features = np.array([
        row['Q1'], 
        row['Q2'], 
        row['Q4'], 
        row['Q8']
    ], dtype=np.float32)
    scaled_numeric = (numeric_features - feature_means) / feature_stds
    
    # Get one-hot encoded features
    q3_features = np.array([row.get(f'Q3_{cat}', 0) for cat in all_q3_categories])
    q7_features = np.array([row.get(f'Q7_{cat}', 0) for cat in all_q7_categories])
    
    # Combine all features
    lr_features = np.concatenate([
        scaled_numeric,
        q3_features,
        q7_features
    ])
    
    lr_probs = predict_lr(lr_features)
    
    # Combined prediction
    combined = 0.44*nb_probs + 0.56*lr_probs
    return label_mapping[str(np.argmax(combined))]

# --- Required Interface Function ---
def predict_all(csv_filename):
    """
    Parameters:
        csv_filename (str): Path to CSV file with the same format as training data
    
    Returns:
        list: Predictions in order of rows in the CSV file
    """
    df = pd.read_csv(csv_filename)
    
    required_columns = ['Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)', 'Q2: How many ingredients would you expect this food item to contain?', 'Q4: How much would you expect to pay for one serving of this food item?', 'Q8: How much hot sauce would you add to this food item?', 
                        'Q5: What movie do you think of when thinking of this food item?', 'Q6: What drink would you pair with this food item?', 'Q3: In what setting would you expect this food to be served? Please check all that apply', 'Q7: When you think about this food item, who does it remind you of?']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        
    df = preprocess_data(df)
    
    for cat in all_q3_categories:
        col = f'Q3_{cat}'
        if col not in df.columns:
            df[col] = 0
            
    for cat in all_q7_categories:
        col = f'Q7_{cat}'
        if col not in df.columns:
            df[col] = 0
    
    predictions = []
    for _, row in df.iterrows():
        try:
            pred = predict_single(row)
            predictions.append(pred)
        except Exception as e:
            raise ValueError(f"Error processing row {_}: {str(e)}")
    
    return predictions

# --- Example Usage ---
if __name__ == "__main__":
    # Test prediction
    predictions = predict_all('cleaned_data_combined_modified.csv')
    print(predictions)
    