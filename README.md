![](https://cdn.aarp.net/content/aarpe/en/home/home-family/personal-technology/info-2021/tips-to-use-search-engines/_jcr_content/root/container_main/container_body_main/container_body1/container_body_cf/container_image/articlecontentfragment/cfimage.coreimg.50.932.jpeg/content/dam/aarp/home-and-family/personal-technology/2023/10/1140-search-engine-tips.jpg)
# Legal Document Search System

## Overview

This project is designed to facilitate advanced search functionality within legal documents using PySpark for data processing and Streamlit for the user interface. The system preprocesses the legal opinions, constructs an inverted index, calculates term frequencies, and other relevant metrics to improve the search capabilities.
![](https://github.com/abh2050/searchengine/blob/main/search.png)
## Components

### Data Preprocessing
- **Source Code**: Python with PySpark.
- **Description**: The legal documents are loaded into a Spark DataFrame, where they are tokenized, stopwords are removed, and stemmed using NLTK. This preprocessing is essential for reducing the dimensionality of the dataset and focusing on meaningful words.

### Inverted Index
- **Description**: An inverted index is created to allow fast full-text searches. It lists each word and a list of the documents in which that word appears. This structure is crucial for efficient retrieval of documents during the search process.
- **Implementation Details**: Stored in `inverted_index.parquet`, this DataFrame maps each word to the list of document IDs where the word is found, facilitating quick lookups for any given term.

### Spark DataFrames
- **Implementation Details**: Data is stored in various DataFrame structures, such as:
  - `flat_words_df`
  - `doc_lengths_df`
  - `term_frequencies_df`
  - `idf_df`
  - `scoring_params_df`
  - `opinion_text_df`

These DataFrames are saved as Parquet files to optimize both storage and query performance.

### Search Functionality
- **Source Code**: Streamlit and Python.
- **Features**:
  - Users can input search queries which are then preprocessed.
  - The BM25 scoring algorithm is used to rank documents based on the query.
  - Results are displayed in the Streamlit app, showing the top documents along with snippets of the text.

### BM25 Ranking Algorithm
- **Description**: The BM25 algorithm provides a way to rank documents based on the query terms appearing in each document. It considers term frequency, inverse document frequency, and the length of the document to provide a relevancy score.
- **Implementation Details**: Scores are calculated dynamically based on the user's query. Each document's relevance to the query is computed, allowing the most pertinent documents to be ranked higher.

### Saving DataFrames
DataFrames are saved to individual Parquet files for persistent storage, with paths relative to the project directory. This allows for efficient data retrieval and manipulation in subsequent sessions.

## Usage

To use the search system, run the Streamlit application script. This script loads the preprocessed data, processes user queries, and calculates document scores to present the search results effectively.

## Streamlit Interface
The application provides a simple and intuitive interface for entering search queries and viewing results. Users can adjust settings such as the length of text snippets displayed.

## Link to Application
Visit the application at [Legal Document Search App](https://searchenginespark.streamlit.app/).
