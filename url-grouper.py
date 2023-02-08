import os
import re
import csv
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def get_text_from_url(url):
    """
    Returns the text within the <article> tag of a given URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article')
        if article:
            return article.text
        else:
            return ""
    except:
        return ""

def preprocess_text(text):
    """
    Preprocesses text data by removing punctuation, numbers, and special characters.
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text

def get_similarity_matrix(documents):
    """
    Computes the cosine similarity matrix of a list of documents.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def cluster_documents(similarity_matrix, threshold):
    """
    Clusters documents based on the cosine similarity matrix and a similarity threshold.
    """
    cluster_number = 0
    clusters = []
    visited = [False] * len(similarity_matrix)

    for i in range(len(similarity_matrix)):
        if visited[i]:
            continue

        cluster = []
        dfs(i, visited, cluster, similarity_matrix, threshold)
        clusters.append(cluster)
        cluster_number += 1

    return clusters

def dfs(i, visited, cluster, similarity_matrix, threshold):
    """
    Depth-first search for finding clusters.
    """
    visited[i] = True
    cluster.append(i)

    for j in range(len(similarity_matrix)):
        if visited[j]:
            continue

        if similarity_matrix[i][j] >= threshold:
            dfs(j, visited, cluster, similarity_matrix, threshold)

if __name__ == "__main__":
    # Load the list of URLs from a text file
    with open("urls.txt", "r") as file:
        urls = file.read().splitlines()

    # Get the text content of each URL and preprocess it
    documents = [preprocess_text(get_text_from_url(url)) for url in urls]

    # Compute the cosine similarity matrix of the documents
    similarity_matrix = get_similarity_matrix(documents)

    # Cluster the documents based on a similarity threshold of 0.5
    clusters = cluster_documents(similarity_matrix, 0.5)

    # Export the results to a CSV
with open("clusters.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Cluster", "URL"])
    
    for cluster_index, cluster in enumerate(clusters):
        for document_index in cluster:
            writer.writerow([cluster_index, urls[document_index]])

