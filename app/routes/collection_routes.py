from flask import Blueprint, request, Response, make_response, abort
from ..db import db
from app.models.collection import Collection
from app.models.paper import Paper
from .route_utilities import validate_model, create_model

bp = Blueprint("bp_collection", __name__, url_prefix="/collections")

@bp.post("")
def create_collection():
    request_body = request.get_json()
    return create_model(Collection, request_body)


@bp.post("/<collection_id>/papers")
def create_paper_on_collection(collection_id):
    collection = validate_model(Collection, collection_id)
    request_body = request.get_json()

    # Only require title, source, and URL
    required_fields = ["title", "source", "URL"]
    for field in required_fields:
        if field not in request_body or request_body[field] == "":
            abort(400, description=f"Missing required field: {field}")

    request_body["collection_id"] = collection.collection_id

    response, status_code = create_model(Paper, request_body)
    return make_response(response, status_code)



@bp.get("")
def get_all_collections():
    query = db.select(Collection).order_by(Collection.collection_id)
    collections = db.session.scalars(query)

    collections_response = []
    for collection in collections:
        collection_dict = collection.to_dict()
        collection_dict["papers_count"] = len(collection.papers)
        collection_dict["description"] = collection.description  # Optional if already in to_dict()
        collections_response.append(collection_dict)

    return collections_response


@bp.get("/<collection_id>")
def get_one_collection_by_id(collection_id):
    collection = validate_model(Collection, collection_id)
    return collection.to_dict()


@bp.get("/<collection_id>/papers")
def get_all_papers_on_collection(collection_id):
    collection = validate_model(Collection, collection_id)

    collection_papers = [paper.to_dict() for paper in collection.papers]
    response = {
        "collection_id": collection.collection_id,
        "title": collection.title,
        "owner": collection.owner,
        "description": collection.description,  # <-- add this
        "papers": collection_papers
    }

    return make_response(response, 200)


@bp.delete("/<collection_id>")
def delete_collection(collection_id):
    collection = validate_model(Collection, collection_id)
    
    papers_to_delete = Paper.query.filter_by(collection_id=collection_id).all()
    
    for paper in papers_to_delete:
        db.session.delete(paper)

    db.session.delete(collection)
    db.session.commit()

    return Response(status=204, mimetype="application/json")


@bp.delete("/<paper_id>")
def delete_paper(paper_id):
    paper = validate_model(Paper, paper_id)
    db.session.delete(paper)
    db.session.commit()

    return Response(status=204, mimetype="application/json")


# @bp.post("/<collection_id>/papers")
# def create_paper_on_collection(collection_id):

#     collection = validate_model(Collection, collection_id)

#     request_body = request.get_json()
#     paper_ids_list = request_body.get("paper_ids")

#     if not paper_ids_list or not isinstance(paper_ids_list, list):
#         response = {"message": f"Invalid request"}
#         abort(make_response(response, 400))

#     updated_papers = []

#     for id in paper_ids_list:
#         paper = validate_model(Paper, id)
#         updated_papers.append(paper)
    
#     collection.papers = updated_papers
#     db.session.commit()

#     response = {
#         "id": collection.collection_id,
#         "paper_ids" : [paper.paper_id for paper in updated_papers]
#     }

#     return make_response(response, 200)
