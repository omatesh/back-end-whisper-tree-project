from flask import Blueprint, abort, make_response, request, Response
from app.models.paper import Paper
from ..db import db
from .route_utilities import validate_model, create_model

bp = Blueprint("bp_paper", __name__, url_prefix="/papers")

@bp.put("/<paper_id>")
def update_paper_likes(paper_id):
    paper = validate_model(Paper, paper_id)
    paper.likes_count += 1
    db.session.commit()
    return {"paper": paper.to_dict()}, 200

@bp.delete("/<paper_id>")
def delete_paper(paper_id):
    paper = validate_model(Paper, paper_id)
    db.session.delete(paper)
    db.session.commit()
    return Response(status=204, mimetype="application/json")

@bp.patch("/<paper_id>")
def update_paper_fields(paper_id):
    paper = validate_model(Paper, paper_id)
    data = request.get_json()

    if "title" in data:
        paper.title = data["title"]
    if "abstract" in data:
        paper.abstract = data["abstract"]
    if "authors" in data:
        paper.authors = data["authors"]
    if "publicationDate" in data:
        paper.publicationDate = data["publicationDate"]
    if "source" in data:
        paper.source = data["source"]
    if "URL" in data:
        paper.URL = data["URL"]

    db.session.commit()
    return {"paper": paper.to_dict()}, 200

# @bp.put("/<paper_id>")
# def update_paper(paper_id):
#     paper = validate_model(Paper, paper_id)
#     request_body = request.get_json()
#
#     paper.message = request_body["message"]
#     paper.likes_count = request_body["likes_count"]
#     paper.collection_id = request_body["collection_id"]
#     db.session.commit()
#
#     return Response(status=204, mimetype="application/json")