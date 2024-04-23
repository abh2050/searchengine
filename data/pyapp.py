import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Assuming preprocessed data are stored in relative paths in the data folder
path_to_flat_words = "./data/flat_words.parquet"
path_to_doc_lengths = "./data/doc_lengths.parquet"
path_to_term_frequencies = "./data/term_frequencies.parquet"
path_to_idf_values = "./data/idf_values.parquet"
path_to_scoring_params = "./data/scoring_params.parquet"
path_to_opinion_texts = "./data/opinion_text.parquet"

# Load preprocessed data using pandas
@st.cache_resource
def load_preprocessed_data():
    flat_words_df = pd.read_parquet(path_to_flat_words)
    doc_lengths_df = pd.read_parquet(path_to_doc_lengths)
    term_frequencies_df = pd.read_parquet(path_to_term_frequencies)
    idf_df = pd.read_parquet(path_to_idf_values)
    scoring_params_df = pd.read_parquet(path_to_scoring_params)
    opinion_texts_df = pd.read_parquet(path_to_opinion_texts)
    return flat_words_df, doc_lengths_df, term_frequencies_df, idf_df, scoring_params_df, opinion_texts_df

# Get preprocessed data
flat_words_df, doc_lengths_df, term_frequencies_df, idf_df, scoring_params_df, opinion_texts_df = load_preprocessed_data()

# Get avgdl value from scoring_params_df
avgdl = scoring_params_df.iloc[0]["avgdl"]

# Define text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
    return words

# Define function for preprocessing queries
def preprocess_query(query):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Define a function for calculating BM25 scores
def calculate_bm25(term_freq, doc_length, avgdl, idf, k1=1.2, b=0.75):
    term_freq = float(term_freq)
    doc_length = float(doc_length)
    avgdl = float(avgdl)
    idf = float(idf)
    return idf * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_length / avgdl)))

# Streamlit UI
def main():
    st.title("Legal Document Search")
    query = st.text_input("Enter your search query:")
    
    # Slider to control the maximum length of the opinion text displayed
    max_text_length = st.slider("Max length of opinion text", min_value=100, max_value=5000, value=1000, step=100)

    if query:
        query_terms = preprocess_query(query)
        st.write("Preprocessed query terms:", query_terms)

        # Filter term_frequencies_df for the query terms
        filtered_term_freqs_df = term_frequencies_df[term_frequencies_df["word"].isin(query_terms)]
        term_freqs_idf_df = filtered_term_freqs_df.merge(idf_df, on="word", how="left")
        term_freqs_idf_lengths_df = term_freqs_idf_df.merge(doc_lengths_df, on="doc_id", how="left")

        # Calculate BM25 score for each term-document pair
        term_freqs_idf_lengths_df["bm25_score"] = term_freqs_idf_lengths_df.apply(
            lambda row: calculate_bm25(row["term_freq"], row["doc_length"], avgdl, row["idf"]),
            axis=1,
        )

        # Aggregate scores by document
        result_df = term_freqs_idf_lengths_df.groupby("doc_id")["bm25_score"].sum().reset_index(name="total_score")

        # Display top N documents
        top_docs = result_df.nlargest(10, "total_score")

        st.subheader("Top Search Results")
        for _, doc in top_docs.iterrows():
            doc_id = doc["doc_id"]
            score = doc["total_score"]
            # Fetch the opinion text for the current document ID
            opinion_text = opinion_texts_df[opinion_texts_df["doc_id"] == doc_id]["opinion_text"].values[0]
            
            # Trim the opinion text to the user-selected length
            displayed_text = opinion_text[:max_text_length] + "..." if len(opinion_text) > max_text_length else opinion_text
            st.write(f"Document ID: {doc_id}, BM25 Score: {score}")
            st.write(f"Opinion Text: {displayed_text}")
            st.write("---")

if __name__ == "__main__":
    main()
