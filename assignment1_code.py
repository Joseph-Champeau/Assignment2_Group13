import re

import numpy as np
import itertools as it

from collections import defaultdict
from nltk.stem import PorterStemmer

# PREPROCESS SECTION

# Load stopwords
with open('./StopWords.txt', 'r') as f:
    stop_words = set(f.read().splitlines())
del f

# Initialize stemmer
stemmer = PorterStemmer()

def preprocess(text: str):
    # Remove non-alphanumerical or space characters
    clean = re.sub(r'[^\w\s]', ' ', text)

    # Tokenization and lowercasing
    tokens = re.findall(r'\b\w+\b', clean)

    # Remove numbers
    tokens = [token for token in tokens if not token.isnumeric()]
    
    # Stopword removal
    tokens = [token for token in tokens if token not in stop_words]
    
    # Porter stemming
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# INDEX SECTION

def create_inverted_index(documents):
    """
    Reads each document in documents (an array of pairs (id, text)).
    Returns the inverted index (a dictionary of tokens to pairs (id, num))
    """
    # Initialize inverted index
    inverted_index = defaultdict(list)

    # Process each document
    for doc_id, doc_text in documents:
        tokens = preprocess(doc_text)

        # Sort the tokens so that they can be grouped
        tokens = sorted(tokens)

        # Use groupby to get unique tokens and repetitions
        for token, group in it.groupby(tokens):
            # Get the number of times the term appears in the text
            num_repetitions = len(list(group))
            
            # Add (doc_id, num_repetitions) pair to index
            inverted_index[token].append((doc_id, num_repetitions))
    
    return inverted_index

def search_inverted_index(query_text, inverted_index):
    """
    Returns all documents that contain at least one word in the query tokens
    using the inverted index
    """
    doc_set = set()

    query_tokens = preprocess(query_text)

    for token in query_tokens:
        doc_list = inverted_index.get(token, None)

        # Handle token not existing in vocabulary
        if doc_list == None: continue

        # Add the document id to the result 
        doc_ids = tuple(map(lambda doc_data: doc_data[0], doc_list))
        doc_set.update(doc_ids)

    return list(doc_set)

# RETRIEVAL SECTION

def generate_doc_matrix(num_docs, inverted_index):
    """
    Generates a document-term frequency matrix.
    Rows are document ids, ordered by the order they are seen
    Columns are terms, ordered by alphabetical order

    num_docs must be the total number of documents.

    Returns tuple (
        doc_id_order: tuple of doc_ids in the order they appear in the matrix
        doc_matrix: numpy 2D array with term frequencies per document
    )
    """

    # Create the empty doc_id_order to keep
    doc_id_order = []

    # Create a temporary dictionary to quickly convert document ids to indices
    doc_id_lookup = {}
    
    # Create the document-term weight matrix and initialise it to zeroes
    doc_matrix = np.zeros((num_docs, len(inverted_index)), dtype=int) # Perhaps we could use a sparse matrix somehow?

    # Loop over all terms in alphabetical order
    for term_index, term in enumerate(sorted(inverted_index)):

        # Iterate over all document hits
        for doc_id, num_rep in inverted_index[term]:

            # Add the document id if this is the first time seeing it
            if doc_id not in doc_id_lookup:
                doc_id_lookup[doc_id] = len(doc_id_order)
                doc_id_order.append(doc_id)
            
            # Get the corresponding row index
            doc_index = doc_id_lookup[doc_id]

            # Set the entry to num_rep
            doc_matrix[doc_index, term_index] = num_rep
            
    del doc_id_lookup

    return tuple(doc_id_order), doc_matrix

def compute_tfidf_params(doc_matrix):
    """
    Computes TF-IDF weights from doc_matrix

    The 2D weight matrix of entire corpus is returned as the first value.
    The second returned value is the scale idf/tf_max
    """
    # Normalize the term frequencies
    tf_max = np.max(doc_matrix, 0)
    tf_matrix = doc_matrix / tf_max

    # Calculate document frequencies
    df = np.where(doc_matrix > 0, 1, 0).sum(0)

    # Calculate inverse IDF
    N = doc_matrix.shape[0] # number of documents
    idf = np.log2(N/df)

    return tf_matrix * idf, idf/tf_max

def compute_query_vector(query_text, vocabulary):
    """
    Given the query text and corpus vocabulary (sorted), generates a corresponding query vector.
    """
    # Create an empty vector to contain the result
    query_vector = np.zeros(shape=(len(vocabulary),), dtype=int)

    # Preprocess the query as if it were a document
    query_tokens = preprocess(query_text)

    # Sort the tokens so that they can be grouped
    query_tokens = sorted(query_tokens)

    # Use groupby to get unique tokens and repetitions
    for token, group in it.groupby(query_tokens):
        # If the token doesn't exist in the corpus, ignore it
        if token not in vocabulary:
            continue

        # Get the index of the token in the vocabulary. Not super efficient but simpler
        token_index = vocabulary.index(token)

        # Get the number of times the term appears in the text
        num_repetitions = len(list(group))
        
        # Set the repetition value in the query vector
        query_vector[token_index] = num_repetitions
    
    return query_vector

def compute_cosine_similarity(tfidf_query, tfidf_docs):
    """
    Computes the cosine similaries of a query TF-IDF vector many document TF-IDF vectors.
    The result is a numpy array with a size corresponding to the number of compared documents.
    """

    query_norm = np.linalg.norm(tfidf_query)
    doc_norms = np.linalg.norm(tfidf_docs, axis=1)

    # Initialise to zero
    result = np.zeros(shape=(len(tfidf_docs)))

    # Handle a zero-query
    if query_norm == 0: return result

    # Handle zero-documents
    non_zero_docs = doc_norms>0

    # Cosine similarity is dot product of document and query norms, divided by the norms
    result[non_zero_docs] = (tfidf_docs[non_zero_docs] @ tfidf_query) / (query_norm * doc_norms)

    return result

# Main function used to evaluate a set of queries (list of strings)
# This is a generator that returns query results 1 by 1
def evaluate_queries(num_docs, inverted_index, queries, topn=100):
    """
    Evaluates the topn top documents for the specified query.

    This is a generator that returns iterators of the form (doc_id, score)
    """
    # Compute document matrix and get sorted vocabulary
    doc_id_order, doc_matrix = generate_doc_matrix(num_docs, inverted_index)
    vocabulary = sorted(inverted_index)

    # Get TF-IDF weight matrix and parameters for corpus
    corpus_weight_matrix, tf_idf_scale = compute_tfidf_params(doc_matrix)

    # Process each query
    for query_text in queries:
        # Compute a query vector
        query_vector = compute_query_vector(query_text, vocabulary)

        # Compute TF-IDF of query
        query_weights = query_vector * tf_idf_scale

        # Fetch relevant documents to the query
        relevant_documents = search_inverted_index(query_text, inverted_index)
        relevant_document_ids = list(map(
            lambda doc_id: doc_id_order.index(doc_id),
            relevant_documents
        ))

        if len(relevant_documents) == 0:
            yield zip([], [])
            continue

        # Calculate the cosine similarities
        cosine_similarities = compute_cosine_similarity(
            query_weights,
            corpus_weight_matrix[relevant_document_ids] # Filter only for relevant documents
        )

        # Sort the cosine similarities and get the top n scores
        top_n_temp = np.argsort(cosine_similarities)[-topn:][::-1]
        top_n_scores = cosine_similarities[top_n_temp]
        top_n_scores_indices = np.array(relevant_document_ids)[top_n_temp]
        top_n_docs_ids = np.array(doc_id_order)[top_n_scores_indices]

        # Return the result
        yield zip(top_n_docs_ids.tolist(), top_n_scores.tolist()), len(relevant_documents)
    
    return None