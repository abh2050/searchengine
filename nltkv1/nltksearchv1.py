#!/usr/bin/env python
# coding: utf-8

# # Legal Document Search Project
# 
# Welcome to the Legal Document Search Project. This project aims to provide an efficient and effective way to search through a large corpus of legal documents. 
# 
# We use Natural Language Processing (NLP) techniques, specifically BM25 (Best Matching 25) algorithm, to index and rank documents based on a given query. 
# 
# The project also includes a simple user interface built with Streamlit, allowing users to input their search queries and view the top matching documents.
# 
# The project uses several Python libraries, including NLTK for text processing, Polars for data manipulation, and Rank-BM25 for search model generation and querying. 
# 
# The project is organized as a Python Notebook, with each cell representing a step in the data processing and search pipeline. 
# 
# Please follow along with the cells in the notebook to understand the workflow of the project, section headers and other markdown are provided to understand what is going on with the code.
# 

# # Install dependencies
# You will likely need to restart your kernel after installation   
# \**This does not need to be run twice.*

# In[1]:


# get_ipython().run_line_magic('pip', 'install polars')
# get_ipython().run_line_magic('pip', 'install nltk')
# get_ipython().run_line_magic('pip', 'install rank-bm25')
# get_ipython().run_line_magic('pip', 'install scikit-learn')
# get_ipython().run_line_magic('pip', 'install streamlit')
# get_ipython().run_line_magic('pip', 'install streamlit_jupyter')


# # Initialize NLTK and DataFrame

# In[3]:


import polars as pl
df = pl.read_ndjson('/Users/abhishekshah/Desktop/Legal project/archive/text.data.jsonl')
df = df.slice(int(-0.5 * len(df)))
df.write_parquet('legal.corpus.par')


# In[4]:


import polars as pl
import nltk
from pathlib import Path

# Ensure NLTK data is downloaded
nltk.download('stopwords')
print('\'stopwords\' corpus downloaded')
nltk.download('punkt')
print('\'punkt\' tokenizer downloaded')


# DATA DIRECTORIES
corpus_dir = Path('legal.corpus.par').resolve()
tokens_dir = Path('legal.tokens.par').resolve()

# Load data
documents_df = pl.read_parquet(corpus_dir)
print('\nData loaded to DataFrame:')
print(documents_df)



# ### Schema and example data from corpus

# In[ ]:


documents_df.schema


# In[ ]:


# # Print the first 5 rows
# documents_df.head()


# In[7]:


# Print the structure of the first non-null `casebody` entry
print(documents_df.filter(documents_df['casebody'].is_not_null())['casebody'].to_frame().unnest('casebody').unnest('data').drop('judges').head(1))


# # Process Text
# 
# Extract, tokenize, then initialize engine   
#    
# Please use this section to process all text for the model. If you have issues running this notebook all at once AFTER running this section:
# 1. Restart your kernel
# 2. Start running the next section

# In[8]:


def extract_text_from_casebody(casebody):
    # Check if 'data' and 'opinions' keys exist and are not empty
    if casebody and 'data' in casebody and 'opinions' in casebody['data'] and casebody['data']['opinions']:
        # Concatenate all 'text' fields from each opinion into one string
        return ' '.join([opinion['text'] for opinion in casebody['data']['opinions']])
    return None

# Apply the function to the DataFrame
documents_df = documents_df.with_columns(
    documents_df['casebody'].map_elements(extract_text_from_casebody, return_dtype=pl.String).alias('case_text')
)

# Verify the first few entries of the extracted text
print(documents_df['case_text'].head())

documents_df = documents_df.select(['id', 'name', 'name_abbreviation', 'case_text']).drop_nulls()


# ### Tokenize the documents and save the dictionary.  
#    
# _This one might take a while_

# In[9]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm # Progress bar (need .auto for polars async)

def tokenize_wrapper(text: str):
    tokens = word_tokenize(text)
    return [word for word in tokens if word.lower() not in stopwords.words('english')]

# Tokenize the text and display progress bar
tokenized_text = [tokenize_wrapper(text[0]) for text in tqdm(documents_df.select(pl.col('case_text')).rows(), ncols=100, desc='Tokenizing text: ')]

# del documents_df  # Remove the original DataFrame from memory for now
# print('Tokenization complete')

# # Tokenize the text and display progress bar
# tokenized_text = [word_tokenize(text[0]) for text in tqdm(documents_df.select(pl.col('case_text')).rows(), ncols=100, desc='Tokenizing text: ')
#                     if text[0].lower() not in stopwords.words('english')] # Remove stopwords

del documents_df # Remove the original DataFrame from memory for now
print('Tokenization complete')

# Store the tokenized documents in new parquet file
print('\nSaving tokenized documents to new dataframe... ', end='')
new_df = pl.DataFrame({'tokens': tokenized_text})
print('done')
print('Saving tokenized documents to new parquet file...', end='')
new_df.write_parquet(tokens_dir)
print('done')

print(f'Tokenized documents saved to new parquet file: {tokens_dir}')


# # BM25 search model
# If you have already generated the `legal.tokens.parquet` file, you can run this code with a restarted kernel if your memory needs to clean up.

# In[10]:


from pathlib import Path
from rank_bm25 import BM25Okapi
import polars as pl
import numpy as np
from nltk.tokenize import word_tokenize

# Redefined for use after kernel restart
def extract_text_from_casebody(casebody):
    # Check if 'data' and 'opinions' keys exist and are not empty
    if casebody and 'data' in casebody and 'opinions' in casebody['data'] and casebody['data']['opinions']:
        # Concatenate all 'text' fields from each opinion into one string
        return ' '.join([opinion['text'] for opinion in casebody['data']['opinions']])
    return None

# Load the tokenized documents
tokenized_df = pl.read_parquet(Path('legal.tokens.par').resolve())
print('Tokenized documents loaded')
# Initialize the BM25 model
bm25 = BM25Okapi(tokenized_df['tokens'].to_numpy())
print('BM25 model loaded')
del tokenized_df # memory cleanup

# Reload the original documents
documents_df = pl.read_parquet(Path('legal.corpus.par').resolve())
print('Original documents loaded')
documents_df = ( documents_df.select(pl.col('id'), pl.col('name'), pl.col('name_abbreviation'))
                             .with_columns(documents_df['casebody'].map_elements(extract_text_from_casebody, return_dtype=pl.String).alias('case_text')) )
print('Original documents reformatted')

# Define a function to search documents using BM25
def bm25_search(query) -> tuple[pl.DataFrame, np.ndarray]:
    # Tokenize the query
    tokenized_query = word_tokenize(query.lower())
    # Retrieve scores
    doc_scores = bm25.get_scores(tokenized_query)
    # Sort documents by their score
    sorted_docs_idx = np.argsort(doc_scores)[::-1]
    # Optionally, return top N documents
    top_n_idx = sorted_docs_idx[:10]
    return documents_df[top_n_idx], doc_scores[top_n_idx]

print('BM25 search function defined')


# ## Examples

# In[11]:


# Example search
query = "property rights"
top_docs, top_scores = bm25_search(query)
#convert top docs to dataframe
top_docs = pl.DataFrame(top_docs)
# Print the top documents
top_docs.head()


# In[12]:


print(top_scores)


# In[13]:


# Example search
query = "murder"
top_docs, top_scores = bm25_search(query)
#convert top docs to dataframe
top_docs = pl.DataFrame(top_docs)
# Print the top documents
top_docs.head()


# In[14]:


#show the case text of the top document
print(top_docs['case_text'][0])


# # User interface

# In[ ]:



import ipywidgets as widgets
from IPython.display import display
import textwrap

# Create a text input widget for the search query
query_input = widgets.Text(
    value='',
    placeholder='Type your query here',
    description='Query:',
    disabled=False
)

# Create a button to execute the search
search_button = widgets.Button(
        description='Search',
        disabled=False,
        button_style='',
        tooltip='Click to search',
        icon='search'
    )

# Create a function to handle the search button click event
def on_search_button_clicked(b):
    # Get the query from the input widget
    query = query_input.value
    # Perform the search
    top_docs, top_scores = bm25_search(query)
    for i, row in enumerate(top_docs.head(10).rows()):
        print(f"Document ID:{row[0]}")
        print(f"{row[1]}'\nScore: {top_scores[i]}")
        # Wrap the text exceeding 200 characters
        wrapped_text = textwrap.fill(row[3][:350], width=115)
        print(f"Case Text: {wrapped_text}...")  # Print the first 100 characters of the case text
        print("\n\n")

# Attach the event handler to the button
search_button.on_click(on_search_button_clicked)

# Display the input bar and button on the same line
box = widgets.HBox([query_input, search_button])

# Display the widgets
display(box)


# In[ ]:


# jupyter nbconvert --to script nltksearchv1.ipynb

