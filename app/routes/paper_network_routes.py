from flask import Blueprint, request, jsonify
from ..db import db
from app.models.paper import Paper
from app.models.collection import Collection
from .route_utilities import validate_model, create_model
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime
import uuid
from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the sentence transformer model for embeddings (fallback)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file")

# Simple embedding cache
embedding_cache = {}

# Create blueprint
bp = Blueprint("bp_network", __name__, url_prefix="/network")



@bp.route('/visualize', methods=['POST'])
def map_networks():
    """
    Get papers with titles and abstracts, generate embeddings, reduce to 2D with PCA, return coordinates
    Request: {"papers": [{"title": "...", "abstract": "..."}, ...]}
    """
    try:
        # =========================================================================
        # 1. Takes a collection of research papers (titles + abstracts)
        #
        # =========================================================================

        #parses incoming HTTP request body from JSON format into Python dictionary
        data = request.get_json()
        #safely retrieves 'papers' array with empty list fallback to prevent KeyError exceptions
        papers = data.get('papers', [])
        
        if not papers:
            return jsonify({"error": "No papers provided"}), 400
            
        if len(papers) < 2:
            return jsonify({"error": "Need at least 2 papers for visualization"}), 400
        
        # Extract abstracts for embedding
        abstracts = [paper.get('abstract', '') for paper in papers]
        
        #Opt2
        # abstracts = []
        # for paper in papers:
        #     abstract = paper.get('abstract', '')
        #     abstracts.append(abstract)

        # =========================================================================
        # 2. EMBEDDING GENERATION
        # Technology: Google Gemini Embedding API (primary), with in-memory caching system
        ''' This is a smart caching and backup system that converts research paper text into 
        mathematical numbers (embeddings) that computers can understand and compare.
        Lazy evaluation with multi-tier fallback and 
        result memoization for embedding vector generation.

       Gemini Embedding Generation. Metod used: embed_content() 
       https://ai.google.dev/gemini-api/docs/embeddings
        embed_content(): Generates embeddings for input text
        - Returns: Dense vector representations of semantic meaning
        - Use case: Similarity comparison, clustering, search
        - Output: Numerical vectors capturing text semantics

        Gemini's Process:

        Text Analysis: Reads and understands the input text using advanced neural networks
        Semantic Encoding: Captures meaning, context, relationships, and concepts
        Vector Generation: Converts understanding into exactly 768 numbers
        Optimization: Ensures similar meanings produce similar number patterns

        Gemini response:

        One list per abstract sent
        Each list has exactly 768 numbers
        Numbers range roughly -1 to +1
        Numbers capture the meaning of that abstract
        '''
        # =========================================================================

        # Check cache first
        #Check if embeddings were previously computed to avoid expensive recalculation
        cached_embeddings = []
        cache_hits = 0
        for abstract in abstracts:
            if abstract in embedding_cache:
                cached_embeddings.append(embedding_cache[abstract])
                cache_hits += 1
            else:
                cached_embeddings.append(None)
        
        # If all abstracts are cached, use cache
        if cache_hits == len(abstracts):
            embeddings = cached_embeddings
            method_used = "cached"
            print(f"Using cached embeddings for all {len(abstracts)} abstracts")
        else:
            # Generate embeddings normally
            embeddings = []
            method_used = "local"
        
            if GEMINI_API_KEY:
                #API Setup and Retry Configuration
                try:
                    method_used = "gemini"
                    print(f"Using Gemini for {len(abstracts)} abstracts")
                    #Initializes API client with authentication credentials
                    client = genai.Client()
                    
                    # Process all abstracts in batch for efficiency
                    retry_delay = 2
                    max_retries = 3
                    
                    # Retry logic:
                    for attempt in range(max_retries):
                        try:
                            #Gemini embedding API Call for text vectorization
                            result = client.models.embed_content(
                                model="gemini-embedding-001", #tells Gemini which AI model to use
                                contents=abstracts,#list of abstracts to convert
                                config=types.EmbedContentConfig(
                                    task_type="SEMANTIC_SIMILARITY",
                                    # Alternative for clustering/grouping
                                    # task_type="CLUSTERING"
                                    
                                    # For search/retrieval systems
                                    # task_type="RETRIEVAL_QUERY"
                                    # task_type="RETRIEVAL_DOCUMENT"
                                    output_dimensionality=768
                                )
                            )
                            
                            # Gemini returns wrapped objects, PCA needs the raw numbers for clustering
                            # Extract embeddings from result
                            embeddings = [embedding.values for embedding in result.embeddings]

                            #Opt 2
                            # embeddings = []
                            # for embedding in result.embeddings:
                            #     embedding_values = embedding.values
                            #     embeddings.append(embedding_values)
                            print(f"Successfully embedded {len(embeddings)} abstracts")
                            break
                            
                        #catches any error that happens during the Gemini API call 
                        # and implements smart retry logic    
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Retry in {retry_delay}s...")
                                time.sleep(retry_delay) #Pauses program for a number of seconds
                                retry_delay *= 2
                            else:
                                print(f"Gemini failed after {max_retries} attempts: {e}")
                                embeddings = []
                                raise e
                #catches setup, connection, API key, configuration Errors:                
                except Exception as e:
                    print(f"Gemini error: {e}")
                    method_used = "local"
            
            # Use local if Gemini failed or not available 
            # The show must go on!
            # Uses local AI model (SentenceTransformer) to convert research abstracts 
            # into number lists, just like Gemini would do, but with lower quality
            if not embeddings:
                print("Using local SentenceTransformer")
                embeddings = model.encode(abstracts).tolist()
                method_used = "local"
            
            # Store new embeddings in cache
            for abstract, embedding in zip(abstracts, embeddings):
                embedding_cache[abstract] = embedding
            print(f"Cached {len(embeddings)} new embeddings")
        
        # Perform K-means clustering using silhouette analysis
        embeddings_array = np.array(embeddings)
        
        # Find optimal number of clusters using silhouette score
        if len(papers) >= 4:  # Need at least 4 points for meaningful clustering
            max_k = min(8, len(papers) // 2)  # Don't exceed half the number of papers
            best_score = -1
            n_clusters = 2
            
            for k in range(2, max_k + 1):
                try:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans_temp.fit_predict(embeddings_array)
                    score = silhouette_score(embeddings_array, labels)
                    
                    if score > best_score:
                        best_score = score
                        n_clusters = k
                        
                except Exception as e:
                    print(f"Silhouette error for k={k}: {e}")
                    continue
            
            print(f"Silhouette analysis: optimal k={n_clusters} (score: {best_score:.3f})")
        else:
            n_clusters = 2  # Default for small datasets
            print(f"Small dataset: using default k={n_clusters}")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
        # PCA reduction to 2D
        pca = PCA(n_components=2, random_state=42)
        coordinates_2d = pca.fit_transform(embeddings_array)
        
        # Generate cluster names using TF-IDF
        def generate_cluster_name_tfidf(cluster_id, paper_indices, all_titles):
            # Get titles for this cluster
            cluster_titles = [papers[i].get('title', '') for i in paper_indices]
            
            if not cluster_titles or len(cluster_titles) == 0:
                return f"Cluster {cluster_id + 1}"
            
            # Combine cluster titles into one document
            cluster_text = ' '.join(cluster_titles)
            
            # Create TF-IDF vectorizer with preprocessing
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),  # Include single words and bigrams
                min_df=1,  # Must appear at least once
                lowercase=True
            )
            
            try:
                # Fit on all titles to get global vocabulary, then transform cluster
                tfidf.fit(all_titles)
                cluster_tfidf = tfidf.transform([cluster_text])
                
                # Get feature names and scores
                feature_names = tfidf.get_feature_names_out()
                scores = cluster_tfidf.toarray()[0]
                
                # Get top scoring terms
                top_indices = scores.argsort()[-3:][::-1]  # Top 3 terms, descending
                top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                if len(top_terms) >= 2:
                    return f"{top_terms[0].title()} & {top_terms[1].title()}"
                elif len(top_terms) == 1:
                    return f"{top_terms[0].title()} Research"
                else:
                    return f"Topic {cluster_id + 1}"
                    
            except Exception as e:
                print(f"TF-IDF naming error for cluster {cluster_id}: {e}")
                return f"Cluster {cluster_id + 1}"
        
        # Group papers by cluster
        cluster_groups = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(i)
        
        # Generate cluster names using TF-IDF
        all_titles = [paper.get('title', '') for paper in papers]
        cluster_names = {}
        for cluster_id, paper_indices in cluster_groups.items():
            cluster_names[cluster_id] = generate_cluster_name_tfidf(cluster_id, paper_indices, all_titles)
        
        # Prepare visualization data
        points = []
        for i, abstract in enumerate(abstracts):
            paper = papers[i]
            title = paper.get("title", "")
            cluster_id = int(cluster_labels[i].item())
            points.append({
                "id": i,
                "title": title,
                "text": title[:15] + "..." if len(title) > 15 else title,
                "full_text": abstract,
                "x": float(coordinates_2d[i][0]),
                "y": float(coordinates_2d[i][1]),
                "cluster_id": cluster_id,
                "cluster_name": cluster_names[cluster_id]
            })
        
        # Prepare cluster summary
        clusters = []
        for cluster_id, name in cluster_names.items():
            cluster_points = [p for p in points if p["cluster_id"] == cluster_id]
            clusters.append({
                "id": int(cluster_id),
                "name": name,
                "size": len(cluster_points),
                "color": ["blue", "red", "green", "orange", "purple"][int(cluster_id) % 5]
            })
        
        return jsonify({
            "points": points,
            "count": len(points),
            "method": method_used,
            "pca_variance": float(sum(pca.explained_variance_ratio_)),
            "clusters": clusters,
            "n_clusters": n_clusters,
            "bounds": {
                "min_x": float(coordinates_2d[:, 0].min()),
                "max_x": float(coordinates_2d[:, 0].max()),
                "min_y": float(coordinates_2d[:, 1].min()),
                "max_y": float(coordinates_2d[:, 1].max())
            }
        }), 200
        
    except Exception as e:
        print(f"Error in map_networks: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/collection-analysis', methods=['POST'])
def collection_analysis():
    """
    Analyze user abstract against papers grouped by collections
    Request: {"user_abstract": "...", "user_title": "..."}
    """
    try:
        data = request.get_json()
        user_abstract = data.get('user_abstract', '')
        user_title = data.get('user_title', 'User Input')
        
        if not user_abstract:
            return jsonify({"error": "User abstract is required"}), 400
        
        # Get all papers with collections from database
        papers = Paper.query.filter(
            Paper.abstract.isnot(None),
            Paper.abstract != '',
            db.func.length(Paper.abstract) > 50,
            Paper.collection_id.isnot(None)
        ).all()
        
        if len(papers) < 2:
            return jsonify({"error": "Need at least 2 papers with collections for analysis"}), 400
        
        # Prepare papers data including user input
        all_papers_data = [{
            "title": user_title,
            "abstract": user_abstract,
            "collection_id": None,  # User input has no collection
            "is_user_input": True
        }]
        
        for paper in papers:
            all_papers_data.append({
                "title": paper.title,
                "abstract": paper.abstract,
                "collection_id": paper.collection_id,
                "is_user_input": False,
                "paper_id": paper.paper_id
            })
        
        # Extract abstracts for embedding
        abstracts = [paper_data['abstract'] for paper_data in all_papers_data]
        
        # Check cache first
        cached_embeddings = []
        cache_hits = 0
        for abstract in abstracts:
            if abstract in embedding_cache:
                cached_embeddings.append(embedding_cache[abstract])
                cache_hits += 1
            else:
                cached_embeddings.append(None)
        
        # Generate embeddings for missing ones
        if cache_hits == len(abstracts):
            embeddings = cached_embeddings
            method_used = "cached"
            print(f"✅ Using cached embeddings for all {len(abstracts)} abstracts")
        else:
            embeddings = []
            method_used = "local"
        
            if GEMINI_API_KEY:
                try:
                    method_used = "gemini"
                    print(f"Using Gemini for {len(abstracts)} abstracts")
                    client = genai.Client()
                    
                    retry_delay = 2
                    max_retries = 3
                    
                    for attempt in range(max_retries):
                        try:
                            result = client.models.embed_content(
                                model="gemini-embedding-001",
                                contents=abstracts,
                                config=types.EmbedContentConfig(
                                    task_type="SEMANTIC_SIMILARITY",
                                    output_dimensionality=768
                                )
                            )
                            
                            embeddings = [embedding.values for embedding in result.embeddings]
                            print(f"✅ Successfully embedded {len(embeddings)} abstracts")
                            break
                            
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"⏳ Retry in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                print(f"❌ Gemini failed after {max_retries} attempts: {e}")
                                embeddings = []
                                raise e
                                
                except Exception as e:
                    print(f"Gemini error: {e}")
                    method_used = "local"
            
            # Store new embeddings in cache
            for abstract, embedding in zip(abstracts, embeddings):
                embedding_cache[abstract] = embedding
            print(f"💾 Cached {len(embeddings)} new embeddings")
        
        # PCA reduction to 2D
        embeddings_array = np.array(embeddings)
        pca = PCA(n_components=2, random_state=42)
        coordinates_2d = pca.fit_transform(embeddings_array)
        
        # Group papers by collection (collections become "clusters")
        collection_groups = {}
        for i, paper_data in enumerate(all_papers_data):
            collection_id = paper_data['collection_id']
            
            if paper_data['is_user_input']:
                # User input gets special treatment - no collection
                continue
                
            if collection_id not in collection_groups:
                collection_groups[collection_id] = []
            collection_groups[collection_id].append(i)
        
        # Get actual collection names from database
        def get_collection_name(collection_id):
            try:
                collection = Collection.query.get(collection_id)
                return collection.title if collection else f"Collection {collection_id}"
            except Exception as e:
                print(f"Error getting collection name for {collection_id}: {e}")
                return f"Collection {collection_id}"
        
        # Generate collection names using actual database names
        collection_names = {}
        for collection_id in collection_groups.keys():
            collection_names[collection_id] = get_collection_name(collection_id)
        
        # Prepare visualization data
        points = []
        for i, paper_data in enumerate(all_papers_data):
            if paper_data['is_user_input']:
                # User input as grey star
                user_point = {
                    "id": i,
                    "title": paper_data['title'],
                    "text": paper_data['title'][:15] + "..." if len(paper_data['title']) > 15 else paper_data['title'],
                    "full_text": paper_data['abstract'],
                    "x": float(coordinates_2d[i][0]),
                    "y": float(coordinates_2d[i][1]),
                    "cluster_id": -1,  # Special ID for user input
                    "cluster_name": "User Input",
                    "is_user_input": True
                }
                points.append(user_point)
                print(f"🌟 Created user input point: {user_point['title']} at ({user_point['x']:.2f}, {user_point['y']:.2f})")
            else:
                collection_id = paper_data['collection_id']
                points.append({
                    "id": i,
                    "title": paper_data['title'],
                    "text": paper_data['title'][:15] + "..." if len(paper_data['title']) > 15 else paper_data['title'],
                    "full_text": paper_data['abstract'],
                    "x": float(coordinates_2d[i][0]),
                    "y": float(coordinates_2d[i][1]),
                    "cluster_id": int(collection_id),
                    "cluster_name": collection_names.get(collection_id, f"Collection {collection_id}"),
                    "is_user_input": False
                })
        
        # Prepare cluster summary (collections + user input)
        clusters = []
        for collection_id, name in collection_names.items():
            cluster_points = [p for p in points if p["cluster_id"] == collection_id]
            clusters.append({
                "id": int(collection_id),
                "name": name,
                "size": len(cluster_points),
                "color": ["blue", "red", "green", "orange", "purple"][int(collection_id) % 5]
            })
        
        # Add user input as special cluster
        clusters.append({
            "id": -1,
            "name": "User Input",
            "size": 1,
            "color": "gray"
        })
        
        return jsonify({
            "points": points,
            "count": len(points),
            "method": method_used,
            "pca_variance": float(sum(pca.explained_variance_ratio_)),
            "clusters": clusters,
            "n_clusters": len(collection_groups) + 1,  # +1 for user input
            "bounds": {
                "min_x": float(coordinates_2d[:, 0].min()),
                "max_x": float(coordinates_2d[:, 0].max()),
                "min_y": float(coordinates_2d[:, 1].min()),
                "max_y": float(coordinates_2d[:, 1].max())
            }
        }), 200
        
    except Exception as e:
        print(f"Error in collection_analysis: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/test', methods=['GET'])
def test_network():
    """Test endpoint to verify network routes are working"""
    try:
        # Count papers in database
        paper_count = Paper.query.count()
        papers_with_abstracts = Paper.query.filter(
            Paper.abstract.isnot(None),
            Paper.abstract != '',
            db.func.length(Paper.abstract) > 50
        ).count()
        
        return jsonify({
            "status": "Network routes loaded successfully",
            "total_papers": paper_count,
            "papers_with_abstracts": papers_with_abstracts,
            # "model_loaded": str(model) if model else "Not loaded"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @bp.route('/related-papers', methods=['POST'])
# def fetch_related_papers():
#     """
#     Fetch related papers from existing database using similarity search
#     """
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         input_type = data.get('input_type', 'abstract')
#         limit = data.get('limit', 50)

#         # Search your existing database for similar papers
#         papers = search_existing_papers(text, limit)
        
#         return jsonify(papers), 200

#     except Exception as e:
#         print(f"Error in fetch_related_papers: {e}")
#         return jsonify({"error": str(e)}), 500

# def search_existing_papers(query_text, limit=50):
#     """Search existing papers in your database using text similarity"""
#     try:
#         # Get all papers with abstracts from database
#         papers = Paper.query.filter(
#             Paper.abstract.isnot(None),
#             Paper.abstract != '',
#             db.func.length(Paper.abstract) > 50
#         ).limit(200).all()  # Get more papers to search through
        
#         if not papers:
#             return []
        
#         # Create text corpus for similarity comparison
#         paper_texts = []
#         paper_data = []
        
#         for paper in papers:
#             # Combine title and abstract for better matching
#             paper_text = f"{paper.title} {paper.abstract}"
#             paper_texts.append(paper_text)
#             paper_data.append(paper)
        
#         # Add query text to corpus
#         all_texts = [query_text] + paper_texts
        
#         # Calculate TF-IDF similarities
#         vectorizer = TfidfVectorizer(
#             max_features=1000, 
#             stop_words='english', 
#             ngram_range=(1, 2)
#         )
#         tfidf_matrix = vectorizer.fit_transform(all_texts)
        
#         # Calculate similarities between query and all papers
#         similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
#         # Sort papers by similarity and get top results
#         paper_similarities = list(zip(paper_data, similarities))
#         paper_similarities.sort(key=lambda x: x[1], reverse=True)
        
#         # Return top similar papers
#         related_papers = []
#         for paper, similarity in paper_similarities[:limit]:
#             if similarity > 0.05:  # Minimum similarity threshold
#                 formatted_paper = {
#                     "paper_id": paper.paper_id,
#                     "title": paper.title,
#                     "abstract": paper.abstract,
#                     "authors": paper.authors,
#                     "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
#                     "source": paper.source,
#                     "URL": paper.url,
#                     "download_url": None,  # Add if you have this field
#                     "core_id": paper.core_id,
#                     "likes_count": paper.likes_count,
#                     "collection_id": paper.collection_id,
#                     "similarity_score": float(similarity)  # Include similarity for debugging
#                 }
#                 related_papers.append(formatted_paper)
        
#         print(f"Found {len(related_papers)} related papers from database")
#         return related_papers
        
#     except Exception as e:
#         print(f"Error searching existing papers: {e}")
#         return []


# @bp.route('/generate', methods=['POST'])
# def generate_paper_network():
#     """
#     Generate network with embeddings, similarities, clustering, and positioning
#     """
#     try:
#         data = request.get_json()
#         input_paper = data.get('input_paper', {})
#         papers = data.get('papers', [])


#         # Create embeddings for all papers
#         all_texts = [input_paper.get('text', '')]
#         all_texts.extend([p.get('abstract', '') + ' ' + p.get('title', '') for p in papers])


#         # Generate embeddings
#         embeddings = model.encode(all_texts)


#         # Calculate similarities
#         similarities = cosine_similarity(embeddings)
#         input_similarities = similarities[0][1:] # Similarities to input paper


#         # Perform clustering
#         n_clusters = min(5, len(papers) // 3 + 1) # Dynamic cluster count
#         if len(papers) > 3:
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#             cluster_labels = kmeans.fit_predict(embeddings[1:]) # Exclude input paper
#         else:
#             cluster_labels = [0] * len(papers)


#         # Generate network nodes
#         nodes = []


#         # Input node
#         input_node = {
#             "id": "input_0",
#             "paper_id": None,
#             "title": input_paper.get('text', '')[:50] + "..." if len(input_paper.get('text', '')) > 50 else input_paper.get('text', ''),
#             "abstract": input_paper.get('text', ''),
#             "authors": None,
#             "publication_date": None,
#             "source": "input",
#             "cluster_id": None,
#             "position": {"x": 0.5, "y": 0.5}, # Center position
#             "similarity": None,
#             "is_input_node": True
#         }
#         nodes.append(input_node)


#         # Paper nodes with circular positioning
#         colors = ["blue", "red", "green", "orange", "purple", "pink", "cyan", "yellow"]
#         clusters = []
#         cluster_info = {}


#         for i, paper in enumerate(papers):
#             cluster_id = f"cluster_{cluster_labels[i]}"


#             # Calculate position (circular layout around input)
#             angle = (2 * np.pi * i) / len(papers)
#             radius = 0.3 + (input_similarities[i] * 0.2) # Closer = more similar
#             x = 0.5 + radius * np.cos(angle)
#             y = 0.5 + radius * np.sin(angle)


#             node = {
#                 "id": f"paper_{paper.get('paper_id', i)}",
#                 "paper_id": paper.get('paper_id'),
#                 "title": paper.get('title', ''),
#                 "abstract": paper.get('abstract', ''),
#                 "authors": paper.get('authors', ''),
#                 "publication_date": paper.get('publication_date', ''),
#                 "source": paper.get('source', ''),
#                 "cluster_id": cluster_id,
#                 "position": {"x": max(0.05, min(0.95, x)), "y": max(0.05, min(0.95, y))},
#                 "similarity": float(input_similarities[i]),
#                 "is_input_node": False
#             }
#             nodes.append(node)

#             # Track cluster info
#             if cluster_id not in cluster_info:
#                 cluster_info[cluster_id] = {
#                     "id": cluster_id,
#                     "label": f"Topic {cluster_labels[i] + 1}",
#                     "description": f"Research cluster {cluster_labels[i] + 1}",
#                     "node_ids": [],
#                     "centroid": embeddings[i + 1].tolist(),
#                     "size": 0,
#                     "color": colors[cluster_labels[i] % len(colors)]
#                 }
#             cluster_info[cluster_id]["node_ids"].append(node["id"])
#             cluster_info[cluster_id]["size"] += 1

#         # Generate edges
#         edges = []
#         edge_threshold = 0.3 # Minimum similarity for edge

#         # Edges from input to papers
#         for i, paper in enumerate(papers):
#             if input_similarities[i] > edge_threshold:
#                 edge = {
#                     "id": f"edge_input_{i}",
#                     "source_id": "input_0",
#                     "target_id": f"paper_{paper.get('paper_id', i)}",
#                     "weight": float(input_similarities[i]),
#                     "edge_type": "semantic_similarity"
#                 }
#                 edges.append(edge)

#         # Edges between papers (if similar enough)
#         for i in range(len(papers)):
#             for j in range(i + 1, len(papers)):
#                 similarity = similarities[i + 1][j + 1]
#                 if similarity > edge_threshold:
#                     edge = {
#                         "id": f"edge_{i}_{j}",
#                         "source_id": f"paper_{papers[i].get('paper_id', i)}",
#                         "target_id": f"paper_{papers[j].get('paper_id', j)}",
#                         "weight": float(similarity),
#                         "edge_type": "semantic_similarity"
#                     }
#                     edges.append(edge)

#         # Create embeddings dictionary
#         embeddings_dict = {}
#         for i, node in enumerate(nodes):
#             embeddings_dict[node["id"]] = embeddings[i].tolist()

#         # Create similarities dictionary
#         similarities_dict = {}
#         for i, node in enumerate(nodes):
#             similarities_dict[node["id"]] = {
#                 other_node["id"]: float(similarities[i][j])
#                 for j, other_node in enumerate(nodes)
#             }

#         # Prepare response
#         network = {
#             "input_paper": {
#                 "text": input_paper.get('text', ''),
#                 "input_type": input_paper.get('input_type', 'topic')
#             },
#             "nodes": nodes,
#             "edges": edges,
#             "clusters": list(cluster_info.values()),
#             "similarities": similarities_dict,
#             "embeddings": embeddings_dict
#         }

#         return jsonify(network), 200

#     except Exception as e:
#             print(f"Error in generate_paper_network: {e}")
#             return jsonify({"error": str(e)}), 500

# UNUSED AI Blueprint - commented out
# ai_bp = Blueprint("bp_ai", __name__, url_prefix="/ai")

# @ai_bp.route('/summarize-paper', methods=['POST'])
# def generate_paper_summary():
#     """
#     Generate AI-powered summary for a paper
#     """
#     try:
#         data = request.get_json()
#         title = data.get('title', '')
#         abstract = data.get('abstract', '')
#         authors = data.get('authors', '')

#         # Simple rule-based summary (replace with AI model)
#         if abstract:
#             # Take first sentence or first 100 characters
#             sentences = abstract.split('. ')
#             summary = sentences[0]
#             if len(summary) > 100:
#                 summary = summary[:97] + "..."
#         else:
#             summary = f"Research by {authors.split(',')[0] if authors else 'Unknown'} on {title[:50]}..."

#         return jsonify({"summary": summary}), 200

#     except Exception as e:
#         print(f"Error in generate_paper_summary: {e}")
#         return jsonify({"error": str(e)}), 500

# @bp.route('/expand', methods=['POST'])
# def expand_network():
#     """
#     Expand network by finding more papers related to a specific node
#     """
#     try:
#         data = request.get_json()
#         node_id = data.get('node_id', '')
#         current_network = data.get('current_network', {})
#         limit = data.get('limit', 20)

#         # Find the node to expand from
#         target_node = None
#         for node in current_network.get('nodes', []):
#             if node['id'] == node_id:
#                 target_node = node
#                 break

#         if not target_node:
#             return jsonify({"error": "Node not found"}), 404

#         # Search for more papers related to this node in your database
#         search_text = target_node.get('title', '') + ' ' + target_node.get('abstract', '')
        
#         # Get existing paper IDs to avoid duplicates
#         existing_paper_ids = set()
#         for node in current_network.get('nodes', []):
#             if node.get('paper_id'):
#                 existing_paper_ids.add(node['paper_id'])
        
#         new_papers = search_existing_papers_excluding(search_text, existing_paper_ids, limit)

#         # Filter out papers already in network
#         existing_titles = set()
#         for node in current_network.get('nodes', []):
#             if node.get('title'):
#                 existing_titles.add(node['title'].lower())

#         filtered_papers = []
#         for paper in new_papers:
#             if paper.get('title', '').lower() not in existing_titles:
#                 filtered_papers.append(paper)

#         # Generate expanded network (similar to generate_paper_network)
#         # This is a simplified version - you may want to maintain the existing network structure
#         expanded_network = generate_expanded_network(current_network, filtered_papers, target_node)

#         return jsonify(expanded_network), 200

#     except Exception as e:
#         print(f"Error in expand_network: {e}")
#         return jsonify({"error": str(e)}), 500
    
# UNUSED Helper functions - commented out

# def search_existing_papers_excluding(query_text, exclude_ids, limit=20):
#     """Search existing papers excluding specified IDs"""
#     try:
#         # Get papers excluding the specified IDs
#         papers = Paper.query.filter(
#             Paper.abstract.isnot(None),
#             Paper.abstract != '',
#             db.func.length(Paper.abstract) > 50,
#             ~Paper.paper_id.in_(exclude_ids) if exclude_ids else True
#         ).limit(100).all()
        
#         if not papers:
#             return []
        
#         # Use same similarity approach as before
#         paper_texts = [f"{paper.title} {paper.abstract}" for paper in papers]
#         all_texts = [query_text] + paper_texts
        
#         vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
#         tfidf_matrix = vectorizer.fit_transform(all_texts)
#         similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
#         # Get top similar papers
#         paper_similarities = list(zip(papers, similarities))
#         paper_similarities.sort(key=lambda x: x[1], reverse=True)
        
#         related_papers = []
#         for paper, similarity in paper_similarities[:limit]:
#             if similarity > 0.1:  # Lower threshold for expansion
#                 formatted_paper = {
#                     "paper_id": paper.paper_id,
#                     "title": paper.title,
#                     "abstract": paper.abstract,
#                     "authors": paper.authors,
#                     "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
#                     "source": paper.source,
#                     "URL": paper.url,
#                     "download_url": None,
#                     "core_id": paper.core_id,
#                     "likes_count": paper.likes_count,
#                     "collection_id": paper.collection_id
#                 }
#                 related_papers.append(formatted_paper)
        
#         return related_papers
        
#     except Exception as e:
#         print(f"Error in search_existing_papers_excluding: {e}")
#         return []

# def search_arxiv(query, limit=25):
#     """Search arXiv API for papers"""
#     try:
#         import feedparser
#         import requests

#         # Clean query for arXiv
#         clean_query = query.replace(' ', '+')
#         url = f"http://export.arxiv.org/api/query?search_query=all:{clean_query}&start=0&max_results={limit}"

#         response = requests.get(url)
#         feed = feedparser.parse(response.content)

#         papers = []
#         for entry in feed.entries:
#             paper = {
#                 'id': entry.id.split('/')[-1],
#                 'title': entry.title,
#                 'abstract': entry.summary,
#                 'authors': ', '.join([author.name for author in entry.authors]),
#                 'date': entry.published,
#                 'source': 'arXiv',
#                 'url': entry.link,
#                 'download_url': entry.link.replace('abs', 'pdf')
#             }
#             papers.append(paper)

#         return papers
#     except Exception as e:
#         print(f"Error searching arXiv: {e}")
#         return []

# def search_core_api(query, limit=25):
#     """Search CORE API for papers (requires API key)"""
#     try:
#         CORE_API_KEY = "YOUR_CORE_API_KEY"

#         if not CORE_API_KEY or CORE_API_KEY == "YOUR_CORE_API_KEY":
#             return [] # Skip if no API key

#         url = "https://api.core.ac.uk/v3/search/works"
#         headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
#         params = {
#             "q": query,
#             "limit": limit
#         }

#         response = requests.get(url, headers=headers, params=params)
#         data = response.json()

#         papers = []
#         for item in data.get('results', []):
#             paper = {
#                 'id': item.get('id'),
#                 'title': item.get('title', ''),
#                 'abstract': item.get('abstract', ''),
#                 'authors': ', '.join([author.get('name', '') for author in item.get('authors', [])]),
#                 'date': item.get('publishedDate', ''),
#                 'source': 'CORE',
#                 'url': item.get('downloadUrl', ''),
#                 'core_id': item.get('id')
#             }
#             papers.append(paper)

#         return papers
#     except Exception as e:
#         print(f"Error searching CORE: {e}")
#         return []

# def generate_expanded_network(current_network, new_papers, anchor_node):
#     """Generate expanded network maintaining existing structure"""
#     # This is a simplified implementation
#     # You should merge new papers with existing network while maintaining positions

#     # For now, return the current network with new papers added
#     # In a full implementation, you'd:
#     # 1. Calculate embeddings for new papers
#     # 2. Find optimal positions for new nodes
#     # 3. Create edges to existing nodes
#     # 4. Update clusters if necessary

#     return current_network
