#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("spark").getOrCreate()


# In[2]:


path_to_data = "/Users/abhishekshah/Desktop/Legal project/small_search/split_file_155.jsonl"
documents_df = spark.read.json(path_to_data)


# In[3]:


#show the schema of the data
documents_df.printSchema()


# In[4]:


#show the first 5 rows of the data
documents_df.show(5)


# In[5]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, lower, regexp_replace, col
from pyspark.sql.types import ArrayType, StringType
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Make sure NLTK datasets are downloaded
nltk.download('punkt')
nltk.download('stopwords')


# In[6]:


# Define stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Adjusted preprocess_text function for Spark UDF compatibility
def preprocess_text(text):
    # Tokenization
    words = word_tokenize(text.lower())
    # Removing stopwords and stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
    return words

# Register the UDF
preprocess_text_udf = udf(preprocess_text, ArrayType(StringType()))


# In[7]:


from pyspark.sql.functions import explode

# Explode the opinions array to work with individual opinions
exploded_docs_df = documents_df.withColumn("opinion", explode("casebody.data.opinions"))

# Apply preprocessing to the 'text' field of each opinion
preprocessed_docs_df = exploded_docs_df.withColumn("words", preprocess_text_udf(col("opinion.text")))

# Show the structure and some example processed data
preprocessed_docs_df.printSchema()
preprocessed_docs_df.select("words").show(truncate=False)


# In[8]:


from pyspark.sql.functions import explode, collect_list

# Verify the structure of the DataFrame
preprocessed_docs_df.printSchema()


# In[9]:


from pyspark.sql.functions import explode, col

# Using the column name 'id' as the document identifier
try:
    # Explode the 'words' column into separate rows and select the 'id' column as the document identifier
    flat_words_df = preprocessed_docs_df.withColumn("word", explode(col("words"))).select(col("id").alias("doc_id"), "word")

    flat_words_df.show(truncate=False)  # For debugging, to see if it works as expected
except Exception as e:
    print(f"Encountered an error: {e}")


# In[10]:


from pyspark.sql.functions import collect_list

# Group by 'word' and aggregate the document IDs into a list
inverted_index_df = flat_words_df.groupBy("word").agg(collect_list("doc_id").alias("doc_ids"))

inverted_index_df.show(truncate=False)


# In[11]:


from pyspark.sql.functions import count

# Assuming 'flat_words_df' contains columns 'doc_id' and 'word'
doc_lengths_df = flat_words_df.groupBy("doc_id").agg(count("word").alias("doc_length"))

doc_lengths_df.show(truncate=False)


# In[12]:


from pyspark.sql.functions import count

# Adjusting the number of partitions to better match the MacBook's capabilities
# The M1 chip has 8 cores, but considering it's a single machine, 
# a lower number might prevent excessive context switching and memory overhead.
# Start with a number like 8 and adjust based on performance and resource utilization.
num_partitions = 8

# Repartition the DataFrame before performing the groupBy operation.
# This can help optimize the parallelism and memory usage.
repartitioned_flat_words_df = flat_words_df.repartition(num_partitions, "doc_id")

# Calculate term frequency (TF) for each word in each document
term_frequencies_df = repartitioned_flat_words_df.groupBy("doc_id", "word").agg(count("word").alias("term_freq"))

# Display the calculated term frequencies. Adjust the number of rows shown as needed.
term_frequencies_df.show(truncate=False)


# In[13]:


from pyspark.sql.functions import log10, countDistinct, col

# Calculate the total number of documents
total_docs = doc_lengths_df.count()

# Since repartition can cause extensive shuffling, especially before a groupBy operation,
# ensure that it's necessary based on your dataset's characteristics and size.
# Adjust the number of partitions if you decide to repartition.
repartitioned_flat_words_df = flat_words_df.repartition(8, "word")

# Calculate the document frequency (DF) for each term: the number of documents containing the term
doc_frequency_df = repartitioned_flat_words_df.groupBy("word").agg(countDistinct("doc_id").alias("doc_freq"))

# Calculate the IDF for each term
idf_df = doc_frequency_df.withColumn("idf", log10(total_docs / col("doc_freq")))

idf_df.show(truncate=False)


# In[14]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_query(query):
    # Convert to lowercase
    query = query.lower()
    
    # Tokenize the query
    tokens = word_tokenize(query)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stem the tokens
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return stemmed_tokens


# In[29]:


from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

# Pre-calculate avgdl
total_doc_length = doc_lengths_df.agg(F.sum("doc_length").alias("total")).collect()[0]["total"]
avgdl = total_doc_length / doc_lengths_df.count()

# Broadcast avgdl and possibly the IDF DataFrame if it's not too large
broadcast_avgdl = spark.sparkContext.broadcast(avgdl)


# In[30]:


def calculate_bm25_spark(query_terms, k1=1.2, b=0.75):
    # Assuming `term_frequencies_df` and `idf_df` are available and registered as tables in Spark SQL
    
    # Construct a DataFrame that contains only the documents and terms of interest
    # This could involve joining `term_frequencies_df` with a broadcasted `idf_df` 
    # and filtering for the relevant terms, then calculating scores in a more bulk SQL operation
    
    # Example assuming necessary joins and filters are applied:
    score_df = spark.sql("""
        SELECT
            t.doc_id,
            SUM(idf.value * (t.term_freq * (1.2 + 1)) / 
                (t.term_freq + 1.2 * (1 - 0.75 + 0.75 * (d.doc_length / {}))))
            AS bm25_score
        FROM term_frequencies AS t
        JOIN document_lengths AS d ON t.doc_id = d.doc_id
        JOIN idf_values AS idf ON t.word = idf.term
        WHERE t.word IN ({})
        GROUP BY t.doc_id
    """.format(broadcast_avgdl.value, ','.join([f"'{term}'" for term in query_terms])))
    
    return score_df


# In[31]:


# Example query
query = "Murder"
query_terms = preprocess_query(query)
print(query_terms)


# In[32]:


# from pyspark.sql.functions import sum as _sum
# # Calculate avgdl for BM25 formula
# total_doc_length = doc_lengths_df.groupBy().agg(_sum("doc_length").alias("total")).collect()[0]["total"]
# avgdl = total_doc_length / doc_lengths_df.count()


# In[33]:


avgdl


# In[34]:


# Apply BM25 scoring to each document that contains at least one query term
# This is a simplification; ideally, you'd filter documents to those containing query terms first
scores = [(row['doc_id'], calculate_bm25_spark(row['doc_id'], query_terms)) for row in doc_lengths_df.collect()]

# Convert to DataFrame for further operations (sorting, filtering)
scores_df = spark.createDataFrame(scores, ["doc_id", "bm25_score"])

# Show top N documents
scores_df.orderBy(col("bm25_score").desc()).show()




