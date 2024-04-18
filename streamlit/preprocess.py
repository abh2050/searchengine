import polars as pl
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from pathlib import Path

def ensure_parquet(corpus_dir: Path) -> Path:
    # Reformat and chop original corpus
    if corpus_dir.suffix == '.jsonl':
        new_path = corpus_dir.parent / 'legal.corpus.par'
        if new_path.exists():
            print("Corpus parquet file already exists.")
            corpus_dir.unlink()
            return new_path
        print("Converting JSONL to Parquet...")
        df = pl.read_ndjson(corpus_dir)
        # Keep only the last 50% of the rows
        df = df.slice(int(-0.5 * len(df)))
        df.write_parquet(new_path)
        # Delete the old jsonl file after conversion
        corpus_dir.unlink()
        return new_path
    return corpus_dir


def preprocess(corpus_dir: Path, tokens_dir: Path):
    if tokens_dir.exists():
        print("Tokenized documents already exist.")
        return
    corpus_dir = ensure_parquet(corpus_dir)
    print("Downloading NLTK data...")
    # Ensure NLTK data is downloaded
    nltk.download('punkt', quiet=True)

    print("Loading corpus data...")
    # Load data
    documents_df = pl.read_parquet(corpus_dir)

    def extract_text_from_casebody(casebody):
        # Check if 'data' and 'opinions' keys exist and are not empty
        if casebody and 'data' in casebody and 'opinions' in casebody['data'] and casebody['data']['opinions']:
            # Concatenate all 'text' fields from each opinion into one string
            return ' '.join([opinion['text'] for opinion in casebody['data']['opinions']])
        return None

    print("Extracting text from casebody...")
    # Apply the function to the DataFrame
    documents_df = documents_df.with_columns(
        documents_df['casebody'].map_elements(extract_text_from_casebody, return_dtype=pl.String).alias('case_text')
    )

    documents_df = documents_df.select(['id', 'name', 'name_abbreviation', 'case_text']).drop_nulls()


    # Tokenize the text and display progress bar
    tokenized_text = [word_tokenize(text[0]) for text in tqdm(documents_df.select(pl.col('case_text')).rows(), ncols=100, desc='Tokenizing text: ')]

    del documents_df # Remove the original DataFrame from memory for now

    print("Storing tokenized documents...")
    # Store the tokenized documents in new parquet file
    new_df = pl.DataFrame({'tokens': tokenized_text})
    new_df.write_parquet(tokens_dir)
    print("Preprocessing completed.")

if __name__ == "__main__":
    preprocess(Path('data/legal.corpus.par'), Path('data/legal.tokens.par'))