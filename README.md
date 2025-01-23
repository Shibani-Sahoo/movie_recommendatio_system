# Movie Recommendation System

## Overview
A movie recommendation system is designed to predict or filter user preferences based on their choices. These systems are widely used across various domains, including movies, music, books, e-commerce, and more. This project focuses on building a machine learning-based recommendation system for movies by leveraging user behavior and movie features.

## Project Objectives
1. **Develop a Recommendation System:** Build a model that suggests movies to users based on their preferences and behavior.
2. **Explore Recommendation Techniques:** Implement various approaches such as collaborative filtering, content-based filtering, and hybrid methods.
3. **Evaluate Model Performance:** Assess the accuracy and effectiveness of the recommendation system using metrics like RMSE (Root Mean Squared Error) and precision.
4. **Enhance User Experience:** Deliver personalized movie recommendations to improve user satisfaction.

## Dataset Description
The dataset used in this project contains information about users, movies, and ratings. The key columns include:

### Users Data
- **UserID:** Unique identifier for each user.
- **Age:** Age group of the user.
- **Gender:** Gender of the user.
- **Occupation:** Occupation of the user.

### Movies Data
- **MovieID:** Unique identifier for each movie.
- **Title:** Name of the movie.
- **Genres:** Genres associated with the movie (e.g., Action, Comedy).

### Ratings Data
- **UserID:** Identifier linking to the user.
- **MovieID:** Identifier linking to the movie.
- **Rating:** Rating given by the user (typically on a scale of 1 to 5).
- **Timestamp:** Time when the rating was given.

## Features of the Recommendation System
1. **Collaborative Filtering:**
   - Leverages user-item interactions to suggest movies based on user behavior.
   - Two types:
     - **User-based Collaborative Filtering:** Finds similar users and recommends movies they have liked.
     - **Item-based Collaborative Filtering:** Recommends movies that are similar to the ones a user has liked.

2. **Content-Based Filtering:**
   - Recommends movies based on the content features (e.g., genres, title keywords).
   - Uses similarity measures like cosine similarity or TF-IDF.

3. **Hybrid Methods:**
   - Combines collaborative and content-based filtering to improve recommendation accuracy.

## Implementation Steps
### 1. Data Loading and Preprocessing
- Load datasets (users, movies, ratings).
- Clean and preprocess data (handle missing values, normalize ratings).
- Merge datasets for a unified structure.

### 2. Exploratory Data Analysis (EDA)
- Visualize user demographics (e.g., age groups, gender distribution).
- Analyze movie popularity based on ratings.
- Identify trends and patterns in user preferences.

### 3. Model Development
- **Collaborative Filtering:**
  - Implement matrix factorization techniques such as Singular Value Decomposition (SVD).
  - Build user-item interaction matrices.

- **Content-Based Filtering:**
  - Use movie metadata (genres, title) for similarity calculations.
  - Apply NLP techniques like TF-IDF and cosine similarity.

- **Hybrid Approach:**
  - Combine collaborative and content-based methods for enhanced recommendations.

### 4. Model Evaluation
- Use metrics like Root Mean Squared Error (RMSE) to evaluate prediction accuracy.
- Compare the performance of different recommendation techniques.

### 5. Deployment
- Create a user-friendly interface to input user preferences and display recommended movies.
- Utilize frameworks like Flask or Streamlit for deployment.

## Example Code Snippets
### Data Loading
```python
import pandas as pd

# Load datasets
users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='MovieID')
data = pd.merge(data, users, on='UserID')
```

### Collaborative Filtering (SVD)
```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Prepare the data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

# Build and evaluate the model
svd = SVD()
cross_validate(svd, data, cv=5, verbose=True)
```

### Content-Based Filtering
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Calculate TF-IDF for genres
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies['Genres'])

# Compute similarity
similarity = cosine_similarity(tfidf_matrix)
```

## Results and Insights
- **Popular Movies:** Identified movies that received the highest ratings.
- **User Clusters:** Grouped users with similar preferences for targeted recommendations.
- **Model Performance:**
  - Collaborative Filtering achieved an RMSE of 0.89.
  - Content-Based Filtering effectively recommended movies with similar genres.

## Future Work
- Incorporate additional features such as user reviews and movie summaries.
- Experiment with deep learning models like Autoencoders for collaborative filtering.
- Implement a real-time recommendation engine.

## Tools and Libraries
- **Python**: Primary programming language.
- **Pandas, NumPy**: Data manipulation and preprocessing.
- **Matplotlib, Seaborn**: Data visualization.
- **Scikit-learn**: Content-based filtering and similarity calculations.
- **Surprise**: Collaborative filtering techniques.
- **Flask/Streamlit**: For deployment.


