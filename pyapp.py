import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, sum as sum_, lit
from pyspark.sql.types import ArrayType, StringType, DoubleType
import os
from pyspark.sql.functions import udf, col, explode, sum as sum_, lit
from pyspark.sql.types import ArrayType, StringType, DoubleType
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')


# Create Spark session
spark = SparkSession.builder.appName("LegalSearch").getOrCreate()

# Assuming preprocessed data are stored in these paths
path_to_flat_words = "/Users/abhishekshah/Desktop/Legal project/small_search/flat_words.parquet"
path_to_doc_lengths = "/Users/abhishekshah/Desktop/Legal project/small_search/doc_lengths.parquet"
path_to_term_frequencies = "/Users/abhishekshah/Desktop/Legal project/small_search/term_frequencies.parquet"
path_to_idf_values = "/Users/abhishekshah/Desktop/Legal project/small_search/idf_values.parquet"
path_to_scoring_params = "/Users/abhishekshah/Desktop/Legal project/small_search/scoring_params.parquet"  # This stores avgdl
path_to_opinion_texts = "/Users/abhishekshah/Desktop/Legal project/small_search/opinion_text.parquet"

# Load preprocessed data
@st.cache_resource
def load_preprocessed_data():
    flat_words_df = spark.read.parquet(path_to_flat_words)
    doc_lengths_df = spark.read.parquet(path_to_doc_lengths)
    term_frequencies_df = spark.read.parquet(path_to_term_frequencies)
    idf_df = spark.read.parquet(path_to_idf_values)
    scoring_params_df = spark.read.parquet(path_to_scoring_params)
    opinion_texts_df = spark.read.parquet(path_to_opinion_texts)
    return flat_words_df, doc_lengths_df, term_frequencies_df, idf_df, scoring_params_df, opinion_texts_df

flat_words_df, doc_lengths_df, term_frequencies_df, idf_df, scoring_params_df, opinion_texts_df = load_preprocessed_data()
avgdl = scoring_params_df.collect()[0]["avgdl"]

# Define the preprocessing function for texts
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
    return words

# UDF for preprocessing query texts
preprocess_text_udf = udf(preprocess_text, ArrayType(StringType()))

# Define the function for preprocessing queries
def preprocess_query(query):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Define the UDF for calculating BM25 scores
@udf(DoubleType())
def calculate_bm25_udf(term_freq, doc_length, avgdl, idf, k1=1.2, b=0.75):
    term_freq = float(term_freq)
    doc_length = float(doc_length)
    avgdl = float(avgdl)
    idf = float(idf)
    return idf * (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_length / avgdl)))


# Define UDFs for query processing and BM25 calculation
@udf(ArrayType(StringType()))
def preprocess_query_udf(query):
    # Assuming you have a function `preprocess_query` defined elsewhere or inline here
    return preprocess_query(query)

# Streamlit UI
def main():
    st.title("Legal Document Search")
    query = st.text_input("Enter your search query:")
    
    # Slider to control the maximum length of the opinion text displayed
    max_text_length = st.slider("Max length of opinion text", min_value=100, max_value=5000, value=1000, step=100)


    if query:
        query_terms = preprocess_query(query)  # Directly use the preprocessing function if it's not too complex
        st.write("Preprocessed query terms:", query_terms)

        # Filter term_frequencies_df for the query terms
        filtered_term_freqs_df = term_frequencies_df.filter(col("word").isin(query_terms))
        term_freqs_idf_df = filtered_term_freqs_df.join(idf_df, on="word", how="left")
        term_freqs_idf_lengths_df = term_freqs_idf_df.join(doc_lengths_df, on="doc_id", how="left")

        # Calculate BM25 score for each term-document pair
        scored_df = term_freqs_idf_lengths_df.withColumn(
            "bm25_score",
            calculate_bm25_udf(col("term_freq"), col("doc_length"), lit(avgdl), col("idf"))
        )
        
        # Aggregate scores by document
        result_df = scored_df.groupBy("doc_id").agg(sum_("bm25_score").alias("total_score"))
        
        # Display top N documents
        top_docs = result_df.orderBy(col("total_score").desc()).limit(10).collect()
        
        st.subheader("Top Search Results")
        for doc in top_docs:
            doc_id = doc["doc_id"]
            score = doc["total_score"]
            # Fetch the opinion text for the current document ID
            opinion_text = opinion_texts_df.filter(opinion_texts_df.doc_id == doc_id).select("opinion_text").collect()[0]["opinion_text"]
            
            # Trim the opinion text to the user-selected length
            displayed_text = opinion_text[:max_text_length] + "..." if len(opinion_text) > max_text_length else opinion_text
            st.write(f"Document ID: {doc_id}, BM25 Score: {score}")
            st.write(f"Opinion Text: {displayed_text}")
            # Assuming you can extract and display the document text or summary here
            st.write("---")

if __name__ == "__main__":
    main()
