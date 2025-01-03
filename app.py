import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load and preprocess data
books = pd.read_csv('BX_Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
books = books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
books.rename(columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher'}, inplace=True)

users = pd.read_csv('BX-Users.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

x = ratings['user_id'].value_counts() > 200
y = x[x].index
ratings = ratings[ratings['user_id'].isin(y)]

book_ratings = ratings.merge(books, on='ISBN')
number_rating = book_ratings.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns={'rating': 'no_of_ratings'}, inplace=True)

final_rating = book_ratings.merge(number_rating, on='title')
final_rating = final_rating[final_rating['no_of_ratings'] >= 50]
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

final_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
final_pivot.fillna(0, inplace=True)

book_sparse = csr_matrix(final_pivot)

# Train the NearestNeighbors model
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Function to recommend books
def recommend_books(book_name):
    if book_name not in final_pivot.index:
        return ["Book not found in the dataset"]
    
    book_id = np.where(final_pivot.index == book_name)[0][0]
    distances, suggestions = model.kneighbors(final_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=10)
    
    recommended_books = [final_pivot.index[suggestion] for suggestion in suggestions[0]]
    return recommended_books

# Streamlit interface
st.title("Books Recommendation System")
st.subheader("The Platform where you can get Recommendations of Books")

# Input field for book name
book_name = st.text_input("Enter the book name:")

# Button to get recommendations
if st.button("Get Recommendations"):
    if book_name:
        recommendations = recommend_books(book_name)
        st.write("Recommended Books:")
        for i, book in enumerate(recommendations, 1):
            st.write(f"{i}. {book}")
    else:
        st.write("Please enter a book name.")
