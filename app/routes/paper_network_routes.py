from flask import Blueprint, request, jsonify
from ..db import db
from app.models.paper import Paper
from app.models.collection import Collection
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file")

# Simple embedding cache
embedding_cache = {}

# Create blueprint
bp = Blueprint("bp_network", __name__, url_prefix="/network")



@bp.route('/analyze-idea', methods=['POST'])
def analyze_idea():
    """
    Analyze user idea against papers in a specific collection using clustering and PCA
    Request: {"collection_id": 1, "user_idea": "text describing research idea"}
    """
    try:
        # =========================================================================
        # 1. Takes a collection of research papers (titles + abstracts)
        #
        # =========================================================================

        #parses incoming HTTP request body from JSON format into Python dictionary
        data = request.get_json()
        #safely retrieves 'collection_id' or None if key doesn't exist, preventing KeyError exceptions
        collection_id = data.get('collection_id')
        #safely retrieves 'user_idea' string with empty string fallback to prevent KeyError exceptions  
        user_idea = data.get('user_idea', '')
        
        if not collection_id:
            return jsonify({"error": "Collection ID is required"}), 400
        if not user_idea:
            return jsonify({"error": "User idea text is required"}), 400
        
        # Get papers from the selected and passed to this route collection
        papers = Paper.query.filter(
            Paper.collection_id == collection_id,
            Paper.abstract.isnot(None),
            Paper.abstract != '',
            db.func.length(Paper.abstract) > 50
        ).all()
        
        if len(papers) == 0:
            return jsonify({"error": "This collection has no papers. Add some papers to perform analysis."}), 400
        elif len(papers) < 2:
            return jsonify({"error": "Need at least 2 papers in collection for analysis"}), 400
        
        # creates a list with the user's idea as the first item
        all_papers_data = [{
            "title": "User Input",
            "abstract": user_idea, # User's research idea text
            "is_user_input": True # Special flag to identify iser input
        }]
        
        # appends each paper from the database to the list above
        for paper in papers:
            all_papers_data.append({
                "title": paper.title,
                "abstract": paper.abstract,
                "is_user_input": False,
                "paper_id": paper.paper_id
            })
        
        # Extract abstracts for embedding (includes user input + all papers)
        abstracts = [paper_data['abstract'] for paper_data in all_papers_data]
        
        # Opt2 | Alternative approach using traditional loop
        # abstracts = []
        # for paper_data in all_papers_data:
        #     abstract = paper_data.get('abstract', '')
        #     abstracts.append(abstract)

        # =========================================================================
        # 2. EMBEDDING GENERATION
        # Technology: Google Gemini Embedding API (primary), with in-memory caching system
        ''' This is a smart caching and backup system that converts research paper abstracts and user input into 
            mathematical numbers (embeddings) that computers can understand and compare.
            Lazy evaluation with multi-tier fallback and 
            result memoization for embedding vector generation.

        Gemini Embedding Generation. Method used: embed_content() 
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
                                contents=abstracts, #list of abstracts to convert
                                config=types.EmbedContentConfig(
                                    task_type="CLUSTERING",  # Using clustering task type
                                    # Alternative task types:
                                    # task_type="SEMANTIC_SIMILARITY","RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"
                                    output_dimensionality=768
                                )
                            )
                            
                            # Gemini returns wrapped objects, PCA needs the raw numbers for clustering
                            # Extract embeddings from result
                            embeddings = [embedding.values for embedding in result.embeddings]
                            
                            #Opt2 | Alternative approach using traditional loop
                            # embeddings = []
                            # for embedding in result.embeddings:
                            #     embedding_values = embedding.values
                            #     embeddings.append(embedding_values)
                            print(f"Successfully embedded {len(embeddings)} abstracts")
                            break
                        
                        # catches any error that happens during the Gemini API call 
                        # and implements smart retry logic 
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Retry in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                print(f"Gemini failed after {max_retries} attempts: {e}")
                                embeddings = []
                                raise e
               
                # catches setup, connection, API key, configuration Errors:                
                except Exception as e:
                    print(f"Gemini error: {e}")
                    method_used = "local"
            
            # Return error if Gemini failed
            if not embeddings:
                return jsonify({"error": "Gemini API failed to generate embeddings"}), 500
            
            # Store new embeddings in cache
            for abstract, embedding in zip(abstracts, embeddings):
                embedding_cache[abstract] = embedding
            print(f"Cached {len(embeddings)} new embeddings")
        

        # =========================================================================
        # 3. does clustering analysis and assigns each paper to a cluster
        """K-means looks at all 768 dimensions simultaneously
            Finds papers that are "close" across all 768 numbers
            Groups them into topic clusters 
            K-means doesn't change original 768D paper data at all. It just produces 
            this list of cluster assignments as a result
            """
        # =========================================================================
        
        # K-means is scikit-learn's clustering algorithm 
        # np.array(embeddings)converts embeddings into the NumPy array, format that scikit-learn requires
        embeddings_array = np.array(embeddings) # Type: NumPy array (matrix) w/768 numbers per paper
        
        # Find optimal number of clusters using silhouette score
        if len(all_papers_data) >= 4: # Need at least 4 points for meaningful clustering
            max_k = min(8, len(all_papers_data) // 2) # Don't exceed half the number of papers
            best_score = -1
            n_clusters = 2
            
            for k in range(2, max_k + 1): # Test different numbers of groups
                try:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10) #setting up the sorter
                    #Takes papers (the 768 numbers for each) and sorts them into k groups based on similarity
                    labels = kmeans_temp.fit_predict(embeddings_array)
                    score = silhouette_score(embeddings_array, labels) #checks if these piles are different enough 
                    
                    if score > best_score: #comparing to saved best score
                        best_score = score
                        n_clusters = k
                        
                except Exception as e:
                    print(f"Silhouette error for k={k}: {e}")
                    continue
            
            print(f"Optimal k={n_clusters} (silhouette score: {best_score:.3f})")


        # the fallback for datasets with less than 4 papers
        else:
            n_clusters = 2
            print(f"Small dataset: using default k={n_clusters}")
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) #creates the clustering tool
        cluster_labels = kmeans.fit_predict(embeddings_array) #runs the clustering and assigns each paper to a group
        
        # =========================================================================
        # 4 PCA reduction to 2D for visualization
        # Takes clustered by K-means papers (768D NumPy array)
        # finds the "most important" 2 directions in 768D space
        # converts each paper from 768 numbers → just 2 numbers (x,y)
        # tries to keep papers that were close in 768D still close in 2D
        # =========================================================================
        

        # Func generate cluster names using TF-IDF
        def generate_cluster_name_tfidf(cluster_id, paper_indices, all_titles):
            # Get titles for this cluster from the papers data (from HTTP response)
            cluster_titles = [all_papers_data[i]['title'] for i in paper_indices]
            
            if not cluster_titles or len(cluster_titles) == 0:
                return f"Cluster {cluster_id + 1}"
            
            # Combine cluster titles into one document
            cluster_text = ' '.join(cluster_titles)
            
            # Create TF-IDF vectorizer with preprocessing
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2), # Include single words and bigrams
                min_df=1, # Must appear at least once
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
                top_indices = scores.argsort()[-3:][::-1] # Top 3 terms, descending
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
        
        # reorganizes the cluster assignments from "paper-to-cluster" format ([0,1,2,0,1]) 
        # into "cluster-to-papers" format ( {0: [papers 0,3], 1: [papers 1,4]}) 
        cluster_groups = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(i)
        
        # Generate cluster names
        all_titles = [paper_data['title'] for paper_data in all_papers_data]
        cluster_names = {}
        for cluster_id, paper_indices in cluster_groups.items():
            cluster_names[cluster_id] = generate_cluster_name_tfidf(cluster_id, paper_indices, all_titles)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=42)
        coordinates_2d = pca.fit_transform(embeddings_array)
        
        # Prepare visualization data by
        # combining each paper's 2D coordinates, cluster assignment, title, abstract, and cluster name 
        # into a single dictionary
        points = []
        for i, paper_data in enumerate(all_papers_data):
            # Give user input special cluster treatment -> separate cluster with ID -1

            cluster_id = int(cluster_labels[i].item())
            cluster_name = cluster_names[cluster_id]
            if paper_data['is_user_input']:
                cluster_id = -1
                cluster_name = "User Input"
            
            point = {
                "id": i,
                "title": paper_data['title'],
                "text": paper_data['title'][:15] + "..." if len(paper_data['title']) > 15 else paper_data['title'],
                "full_text": paper_data['abstract'],
                "x": float(coordinates_2d[i][0]),
                "y": float(coordinates_2d[i][1]),
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "is_user_input": paper_data['is_user_input']
            }
            
            # Add paper_id if it's not user input
            if not paper_data['is_user_input']:
                point["paper_id"] = paper_data['paper_id']
            
            points.append(point)
        
        # counts how many papers are in each cluster and assigning each cluster a color
        clusters = []
        for cluster_id, name in cluster_names.items():
            cluster_points = [p for p in points if p["cluster_id"] == cluster_id]
            clusters.append({
                "id": int(cluster_id),
                "name": name,
                "size": len(cluster_points),
                "color": ["blue", "red", "green", "orange", "purple"][int(cluster_id) % 5]
            })
        
        # Add user input as special cluster
        clusters.append({
            "id": -1,
            "name": "User Input",
            "size": 1,
            "color": "gray"
        })
        
        # returns a JSON response containing all the data: the individual paper points with coordinates, 
        # cluster summaries with colors and names, total counts, and the map boundaries (min/max x,y values)
        return jsonify({
            "points": points,
            "count": len(points),
            "method": method_used,
            "pca_variance": float(sum(pca.explained_variance_ratio_)),
            "clusters": clusters,
            "n_clusters": n_clusters,
            "collection_id": collection_id,
            "bounds": {
                "min_x": float(coordinates_2d[:, 0].min()),
                "max_x": float(coordinates_2d[:, 0].max()),
                "min_y": float(coordinates_2d[:, 1].min()),
                "max_y": float(coordinates_2d[:, 1].max())
            }
        }), 200
        
    except Exception as e:
        print(f"Error in analyze_idea: {e}")
        return jsonify({"error": str(e)}), 500


