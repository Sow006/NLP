# Sentiment Analysis Project: All Modules in One Script

# 1. Install Required Packages
# Uncomment the line below if running in a new environment
# !pip install pandas numpy scikit-learn nltk matplotlib seaborn streamlit

# 2. Import Libraries
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 3. Download NLTK Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 4. Data Loading
# Replace 'IMDB_Dataset.csv' with your dataset path
df = pd.read_csv('IMDB_Dataset.csv')
print("Sample Data:")
print(df.head())

# 5. Data Preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(clean_text)

# 6. Feature Engineering
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['sentiment'].map({'positive':1, 'negative':0})

# 7. Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 9. Visualization
df['sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 10. Optional: Simple Web App with Streamlit
# To run, save this script as app.py and run: streamlit run app.py
import streamlit as st

st.title('Sentiment Analysis App')
user_input = st.text_area('Enter text:')
if st.button('Analyze'):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)
    st.write('Sentiment:', 'Positive' if prediction[0] == 1 else 'Negative')

# 11. Tips for Improvement
# - Try advanced models (SVM, Random Forest, BERT)
# - Use GridSearchCV for hyperparameter tuning
# - Document your workflow and results for reproducibility
