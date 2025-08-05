import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
import json
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score

# Custom transformer for column selection
class ColumnSelector:
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column].fillna('')
    
k = 5  # Typical value, can be adjusted
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 1. Load and prepare data
df = pd.read_excel('data.xlsx')

# 2. Data Cleaning with Multi-Category Support
def split_categories(text):
    return [x.strip() for x in str(text).split(',')] if pd.notna(text) else []

cleaned_df = df[[
    'CLEANED Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)',
    'CLEANED Q2',
    'Q3: In what setting would you expect this food to be served? Please check all that apply',
    'CLEANED Q4',
    'Q5: What movie do you think of when thinking of this food item?',
    'Q6: What drink would you pair with this food item?',
    'Q7: When you think about this food item, who does it remind you of?',
    'CLEANED Q8: numeric encoded',
    'Label'
]].copy()

# Rename columns
cleaned_df.columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5_movie', 'Q6_drink', 
                     'Q7_person', 'Q8_hotsauce', 'target_label']

# Process multi-category columns
for col in ['Q3', 'Q7_person']:
    cleaned_df[col] = cleaned_df[col].apply(split_categories)

# Get all unique categories
all_q3_categories = set()
cleaned_df['Q3'].apply(lambda x: all_q3_categories.update(x))
all_q7_categories = set()
cleaned_df['Q7_person'].apply(lambda x: all_q7_categories.update(x))
print(all_q3_categories, all_q7_categories)

# Create binary columns for each category
for category in all_q3_categories:
    cleaned_df[f'Q3_{category}'] = cleaned_df['Q3'].apply(lambda x: int(category in x))

for category in all_q7_categories:
    cleaned_df[f'Q7_{category}'] = cleaned_df['Q7_person'].apply(lambda x: int(category in x))

with pd.option_context('display.max_columns', None):
    print(cleaned_df.head())

# 3. Train-Test Split
X = cleaned_df.drop(['target_label', 'Q3', 'Q7_person'], axis=1)
y = cleaned_df['target_label']

label_mapping = {'Pizza': 0, 'Shawarma': 1, 'Sushi': 2}  # Map string labels to numbers
y = y.map(label_mapping)  # Convert all labels to numbers
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=42)

# 4. Build the Text Feature Pipeline (EXACTLY as you requested)
text_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('movie', Pipeline([
            ('selector', ColumnSelector("Q5_movie")),
            ('vectorizer', TfidfVectorizer())
        ])),
        ('drink', Pipeline([
            ('selector', ColumnSelector("Q6_drink")),
            ('vectorizer', TfidfVectorizer())
        ]))
    ])),
    ('nb', MultinomialNB(alpha=0.1))
])

# 5. Build Structured Feature Pipeline
numeric_features = ['Q1', 'Q2', 'Q4', 'Q8_hotsauce']
q3_features = [f'Q3_{cat}' for cat in all_q3_categories]
q7_features = [f'Q7_{cat}' for cat in all_q7_categories]

structured_pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
    ], remainder='passthrough')),
    ('lr', LogisticRegression(
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000,
        C=0.1))
])

# 6. Train Both Models

print("Training text model...")
text_pipeline.fit(X_train, y_train)

print("Training structured model...")
structured_pipeline.fit(X_train[numeric_features + q3_features + q7_features], y_train)


for col in ['Q5_movie', 'Q6_drink']:
    X_test[col] = X_test[col].astype(str).fillna('')

# 7. Generate and Combine Predictions
print("\nGenerating predictions...")
text_probs = text_pipeline.predict_proba(X_test)
struct_probs = structured_pipeline.predict_proba(X_test[numeric_features + q3_features + q7_features])

# Combine predictions (weighted average)
combined_probs = (0.42 * text_probs + 0.58 * struct_probs)  # Adjust weights based on validation
final_pred = np.argmax(combined_probs, axis=1)

# # Create directory
# Path('model_assets').mkdir(exist_ok=True)

# def export_models():
#     # 1. Export Logistic Regression
#     lr_model = structured_pipeline.named_steps['lr']
#     np.save('model_assets/lr_weights.npy', lr_model.coef_)
#     np.save('model_assets/lr_intercept.npy', lr_model.intercept_)
    
#     # 2. Export Standardization Parameters
#     scaler = structured_pipeline.named_steps['preprocessor'].named_transformers_['num']
#     np.save('model_assets/feature_means.npy', scaler.mean_)
#     np.save('model_assets/feature_stds.npy', scaler.scale_)
    
#     # 3. Export Naive Bayes
#     nb_model = text_pipeline.named_steps['nb']
#     vectorizer = text_pipeline.named_steps['features'].transformer_list[0][1].named_steps['vectorizer']
    
#     word_counts = {
#         str(cls): {
#             word: nb_model.feature_log_prob_[cls][idx] 
#             for word, idx in vectorizer.vocabulary_.items()
#         } 
#         for cls in range(len(nb_model.classes_))
#     }
#     with open('model_assets/nb_word_counts.json', 'w') as f:
#         json.dump(word_counts, f)
    
#     np.save('model_assets/nb_class_priors.npy', nb_model.class_log_prior_)
    
#     # 4. Export TF-IDF Components
#     with open('model_assets/tfidf_vocab.json', 'w') as f:
#         json.dump(vectorizer.vocabulary_, f)
#     np.save('model_assets/tfidf_idf.npy', vectorizer.idf_)
    
#     # 5. Export Label Mapping
#     label_mapping = {
#         int(i): str(cls) 
#         for i, cls in enumerate(text_pipeline.classes_)
#     }
#     with open('model_assets/label_mapping.json', 'w') as f:
#         json.dump(label_mapping, f, indent=4)

# export_models()
# 8. Evaluate All Models
print("\n=== Model Performance ===")
print("\nNaive Bayes (Text Features Only):")
print(classification_report(y_test, text_pipeline.predict(X_test)))

print("\nLogistic Regression (Structured Features Only):")
print(classification_report(y_test, structured_pipeline.predict(X_test[numeric_features + q3_features + q7_features])))

print("\nCombined Model (Hybrid Approach):")
print(classification_report(y_test, final_pred))