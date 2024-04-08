from pathlib import Path
import polars as pl
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import textwrap
import streamlit as st

# Define the paths
tokenized_path = 'https://drive.google.com/file/d/1kLITMyhh3OBynei3HX0qF67pIshhIuJy/view?usp=sharing'
corpus_path = 'https://drive.google.com/file/d/1cwKsLjcoIvInmRTGhAvxWCXy1qWPvHru/view?usp=sharing'

# Function to load data efficiently
@st.cache_resource
def load_data_efficiently():
    st.write("Loading data...")
    tokenized_df = pl.read_parquet(tokenized_path)
    documents_df = pl.read_parquet(corpus_path)
    bm25 = BM25Okapi(tokenized_df.get_column('tokens').to_numpy())
    return bm25, documents_df

# Display a title and loading message
st.title("Legal Search")
st.write("Please wait while the app loads the initial data. This may take 1-2 minutes.")

# Display a progress bar while loading data
with st.spinner("Loading data..."):
    bm25_model, documents_df = load_data_efficiently()

def bm25_search_with_opinion(query, bm25, documents_df, max_length, top_n=5):
    # Tokenize the query and get BM25 scores
    tokenized_query = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Sort documents based on scores and select the top N
    sorted_docs_idx = np.argsort(doc_scores)[::-1][:top_n]
    mask = pl.Series(np.arange(len(documents_df))).is_in(sorted_docs_idx)
    top_docs = documents_df.filter(mask)
    
    # Add scores to the top documents
    scores_df = pl.DataFrame({'score': doc_scores[sorted_docs_idx]})
    top_docs = pl.concat([top_docs, scores_df], how='horizontal')
    
    # Define a function to extract and format the opinion text
    def extract_and_format_opinion(casebody, query, max_length):
        try:
            opinions = casebody['data']['opinions']
            if opinions:
                opinion_text = opinions[0]['text']
                # Highlight the search word in the opinion text
                highlighted_text = opinion_text.replace(query, f"<mark>{query}</mark>")
                # Trim the opinion text based on the max_length
                trimmed_text = textwrap.shorten(highlighted_text, width=max_length, placeholder="...")
                return trimmed_text
            else:
                return "No opinion text found."
        except (KeyError, IndexError):
            return "No opinion text found."
    
    # Include formatted opinion text in the results
    top_docs = top_docs.select(
        pl.col('*'),
        pl.col('casebody').map_elements(lambda x: extract_and_format_opinion(x, query, max_length), return_dtype=pl.Utf8).alias('opinion_text')
    )
    
    return top_docs

# Streamlit app
def main():
    query = st.text_input("Enter your query:")
    max_length = st.slider("Select the maximum length of the opinion text", min_value=100, max_value=1000, value=500, step=100)
    
    if st.button("Search"):
        top_docs_df = bm25_search_with_opinion(query, bm25_model, documents_df, max_length, top_n=5)
        
        st.write(f"Query: '{query}'")
        st.write("Top documents based on your query:")
        
        # Display the results as a table
        for row in top_docs_df.iter_rows():
            st.write(f"Document ID: {row[0]}")  # Access the 'id' column by index
            st.write(f"Text: {row[-1]}")  # Access the 'score' column by index
            st.markdown(row[-2], unsafe_allow_html=True)  # Access the 'opinion_text' column by index
            st.write("---")

if __name__ == "__main__":
    main()
