from flask import Blueprint, request, Response, make_response, abort, jsonify
from ..db import db
from app.models.collection import Collection
from app.models.paper import Paper
from .route_utilities import validate_model, create_model
import os
import requests
from datetime import datetime

bp = Blueprint("bp_core", __name__, url_prefix="/core")

CORE_API_KEY = os.environ.get("CORE_API_KEY") 
CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works"
CORE_WORK_URL = "https://api.core.ac.uk/v3/works" # Endpoint for fetching a single work by ID

# JSON Response from CORE API example:
# {
#   "results": [
#     {
#       "score": ...
#       "document": {
#             "id": ...,
#             "title": ...,
#             "abstract": ...,
#             "authors": [{"name": ..., "orcid": ...,},{"name": ...,}],
#             "publishedDate": "2023-01-15",
#             "year": 2023,
#             "source": {"id": "journal_123","name": "Journal ...","type": "journal"},
#       }
#       "urls": [{"url": ..., "type": "pdf"},{...}],
#       "doi": "10.1000/xyz.2023.01.15",
#       "fullText": "This is a snippet of the full text if available and requested...",
#       "fieldsOfStudy": ["Computer Science","Medicine","Artificial Intelligence"]
#       }
#     },
#   ]
# }

# {
#   "results": [
#     {
#       "score": 10.5,
#       "document": {
#         "id": "core.ac.uk:123456789",
#         "title": "The Impact of Artificial Intelligence on Healthcare Systems",
#         "abstract": "This paper explores the transformative effects of artificial intelligence (AI) on various aspects of healthcare, including diagnostics, personalized medicine, and administrative efficiency. It discusses both the opportunities and challenges associated with AI integration.",
#         "authors": [
#           {
#             "name": "Alice Johnson",
#             "orcid": "0000-0002-1234-5678"
#           },
#           {
#             "name": "Bob Williams"
#           }
#         ],
#         "publishedDate": "2023-01-15",
#         "year": 2023,
#         "source": {
#           "id": "journal_123",
#           "name": "Journal of Medical AI",
#           "type": "journal"
#         },
#         "urls": [
#           {
#             "url": "https://core.ac.uk/download/pdf/123456789.pdf",
#             "type": "pdf"
#           },
#           {
#             "url": "https://example.com/paper/ai-healthcare",
#             "type": "landingPage"
#           }
#         ],
#         "doi": "10.1000/xyz.2023.01.15",
#         "fullText": "This is a snippet of the full text if available and requested...",
#         "fieldsOfStudy": [
#           "Computer Science",
#           "Medicine",
#           "Artificial Intelligence"
#         ]
#       }
#     },
# ]

@bp.post("/search")
def search_core_papers():
    if not CORE_API_KEY: # Check if the API key is loaded
        abort(500, description="CORE API Key is not configured in environment variables.")

    # parses the JSON data sent by frontend in the body of its POST request 
    # to this backend, converting it into a Python dictionary for this backend to use
    request_body = request.get_json()
    query_text = request_body.get("query")
    limit = request_body.get("limit", 10)
    offset = request_body.get("offset", 0)

    if not query_text:
        abort(400, description="Missing 'query' parameter in request body.")

    headers = {
        "Authorization": f"Bearer {CORE_API_KEY}",
        "Content-Type": "application/json"
    }

    # CORE API v3 uses a POST request with a JSON body for search
    core_request_params = {
        "q": query_text,
        "limit": limit,
        "offset": offset
    }

    try:
        response = requests.post(CORE_SEARCH_URL, headers=headers, json=core_request_params)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        core_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling CORE API: {e}")
        abort(500, description=f"Failed to connect to CORE API: {e}")
    except ValueError as e:
        print(f"Error parsing CORE API response: {e}")
        abort(500, description="Failed to parse CORE API response.")

    papers_from_core = []
    # The structure of CORE API v3 search results for /works is typically under 'results'
    for item in core_data.get("results", []):

        # Map CORE API fields to your Paper model fields
        title = item.get("title", "No Title Available")
        abstract = item.get("abstract", "")
        authors_list = item.get("authors", [])
        authors = ", ".join([author.get("name") for author in authors_list if author.get("name")])

        # CORE API v3 might provide publication date in different formats.
        # Common fields are 'publishedDate' or 'year'. Adjust as necessary.
        publication_date_str = item.get("publishedDate")
        publication_date = None
        if publication_date_str:
            try:
                # Attempt to parse common date formats
                publication_date = datetime.strptime(publication_date_str, "%Y-%m-%dT%H:%M:%S").date()
            except ValueError:
                try:
                    publication_date = datetime.strptime(publication_date_str, "%Y").date()
                except ValueError:
                    pass # Keep as None if parsing fails

        url = item.get("downloadUrl", "") # Get first URL if available
        core_id = item.get("id") # CORE's unique identifier for the paper

        papers_from_core.append({
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "publication_date": publication_date.isoformat() if publication_date else None,
            "download_url": url,
            "core_id": core_id,
            # "likes_count": 0, # Default for new papers from CORE
            # "collection_id": None # This paper is not yet in a collection
        })

    return make_response(jsonify(papers_from_core), 200)

# fetch a single paper by its CORE ID
@bp.get("/papers/<core_id>")
def get_paper_by_core_id(core_id):
    if not CORE_API_KEY:
        abort(500, description="CORE API Key is not configured in environment variables.")

    headers = {
        "Authorization": f"Bearer {CORE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Construct the URL for fetching a single work by its ID
    single_work_url = f"{CORE_WORK_URL}/{core_id}"

    try:
        response = requests.get(single_work_url, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors
        core_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling CORE API for single paper: {e}")
        # If the paper is not found, CORE API might return 404
        if response.status_code == 404:
            abort(404, description=f"Paper with CORE ID '{core_id}' not found on CORE API.")
        abort(500, description=f"Failed to connect to CORE API for single paper: {e}")
    except ValueError as e:
        print(f"Error parsing CORE API response for single paper: {e}")
        abort(500, description="Failed to parse CORE API response for single paper.")

    # The response for a single work is the document itself, not nested under 'results'
    paper_info = core_data # For single work, core_data directly contains the document info

    # Map CORE API fields to your Paper model fields
    title = paper_info.get("title", "No Title Available")
    abstract = paper_info.get("abstract", "")
    authors_list = paper_info.get("authors", [])
    authors = ", ".join([author.get("name") for author in authors_list if author.get("name")])

    publication_date_str = paper_info.get("publishedDate")
    publication_date = None
    if publication_date_str:
        try:
            publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                publication_date = datetime.strptime(publication_date_str, "%Y").date()
            except ValueError:
                pass

    source = paper_info.get("source", {}).get("name", "Unknown Source")
    url = paper_info.get("urls", [{}])[0].get("url", "")
    # The core_id is already available from the path parameter, but we can also confirm from response
    retrieved_core_id = paper_info.get("id")

    # Construct the response dictionary
    response_paper_data = {
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "publication_date": publication_date.isoformat() if publication_date else None,
        "source": source,
        "url": url,
        "core_id": retrieved_core_id,
        "likes_count": 0, # Default, as this is not from your DB
        "collection_id": None # Not part of a collection when fetched directly from CORE
    }

    return make_response(jsonify(response_paper_data), 200)

@bp.post("/papers/<core_id>/add-to-collection")
def add_core_paper_to_collection(core_id):
    """Add a CORE paper to a specific collection"""
    if not CORE_API_KEY:
        abort(500, description="CORE API Key is not configured in environment variables.")

    request_body = request.get_json()
    collection_id = request_body.get("collection_id")
    
    if not collection_id:
        abort(400, description="Missing 'collection_id' parameter in request body.")

    # Validate that the collection exists
    collection = Collection.query.get(collection_id)
    if not collection:
        abort(404, description=f"Collection with ID {collection_id} not found.")

    # Check if paper already exists in database
    existing_paper = Paper.query.filter_by(core_id=core_id).first()
    
    if existing_paper:
        # Paper already exists, check if it's in a different collection
        if existing_paper.collection_id == collection_id:
            return make_response(jsonify({
                "message": "Paper is already in this collection",
                "paper": existing_paper.to_dict()
            }), 200)
        else:
            # Update the collection_id to move it to the new collection
            existing_paper.collection_id = collection_id
            db.session.commit()
            return make_response(jsonify({
                "message": "Paper moved to new collection",
                "paper": existing_paper.to_dict()
            }), 200)

    # Paper doesn't exist, fetch it from CORE API and add to collection
    headers = {
        "Authorization": f"Bearer {CORE_API_KEY}",
        "Content-Type": "application/json"
    }

    single_work_url = f"{CORE_WORK_URL}/{core_id}"

    try:
        response = requests.get(single_work_url, headers=headers)
        response.raise_for_status()
        core_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling CORE API for adding paper: {e}")
        if response.status_code == 404:
            abort(404, description=f"Paper with CORE ID '{core_id}' not found on CORE API.")
        abort(500, description=f"Failed to connect to CORE API: {e}")
    except ValueError as e:
        print(f"Error parsing CORE API response: {e}")
        abort(500, description="Failed to parse CORE API response.")

    paper_info = core_data

    # Parse the paper data
    title = paper_info.get("title", "No Title Available")
    abstract = paper_info.get("abstract", "")
    authors_list = paper_info.get("authors", [])
    authors = ", ".join([author.get("name") for author in authors_list if author.get("name")])

    publication_date_str = paper_info.get("publishedDate")
    publication_date = None
    if publication_date_str:
        try:
            publication_date = datetime.strptime(publication_date_str, "%Y-%m-%d").date()
        except ValueError:
            try:
                publication_date = datetime.strptime(publication_date_str, "%Y").date()
            except ValueError:
                pass

    source = paper_info.get("source", {}).get("name", "Unknown Source")
    url = paper_info.get("urls", [{}])[0].get("url", "")

    # Create new paper in database
    new_paper = Paper(
        title=title,
        abstract=abstract,
        authors=authors,
        publication_date=publication_date,
        source=source,
        url=url,
        likes_count=0,
        collection_id=collection_id,
        core_id=core_id
    )

    try:
        db.session.add(new_paper)
        db.session.commit()
        
        return make_response(jsonify({
            "message": "Paper successfully added to collection",
            "paper": new_paper.to_dict()
        }), 201)
    except Exception as e:
        db.session.rollback()
        print(f"Error adding paper to database: {e}")
        abort(500, description="Failed to add paper to collection.")


