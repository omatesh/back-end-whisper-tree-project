from flask import Blueprint, request, make_response, abort, jsonify
from ..db import db
from app.models.paper import Paper
from .route_utilities import validate_model, create_model

bp = Blueprint("bp_papers", __name__, url_prefix="/papers")

@bp.get("/<int:paper_id>")
def get_paper_by_id(paper_id):
    """Get a single paper by ID"""
    paper = Paper.query.get(paper_id)
    if not paper:
        abort(404, description=f"Paper with ID {paper_id} not found.")
    
    return make_response(jsonify(paper.to_dict()), 200)

@bp.put("/<int:paper_id>")
def like_paper(paper_id):
    """Increment the likes count for a paper"""
    paper = Paper.query.get(paper_id)
    if not paper:
        abort(404, description=f"Paper with ID {paper_id} not found.")
    
    try:
        paper.likes_count += 1
        db.session.commit()
        
        return make_response(jsonify({
            "message": "Paper liked successfully",
            "paper": paper.to_dict()
        }), 200)
        
    except Exception as e:
        db.session.rollback()
        print(f"Error liking paper: {e}")
        abort(500, description="Failed to like paper.")

@bp.delete("/<int:paper_id>")
def delete_paper(paper_id):
    """Delete a paper"""
    print(f"🗑️ [BACKEND] Delete request for paper ID: {paper_id}")
    print(f"🗑️ [BACKEND] Paper ID type: {type(paper_id)}")

    paper = Paper.query.get(paper_id)
    if not paper:
        print(f"❌ [BACKEND] Paper with ID {paper_id} not found in database")
        
        # Debug: Show what papers DO exist
        all_papers = Paper.query.all()
        print(f"🔍 [BACKEND] Available papers in database:")
        for p in all_papers:
            print(f"  ID: {p.paper_id}, Title: {p.title}")

        abort(404, description=f"Paper with ID {paper_id} not found.")
    
    print(f"✅ [BACKEND] Found paper: {paper.title}")
    try:
        db.session.delete(paper)
        db.session.commit()
        
        return make_response(jsonify({
            "message": "Paper deleted successfully"
        }), 200)
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting paper: {e}")
        abort(500, description="Failed to delete paper.")


# from flask import Blueprint, abort, make_response, request, Response
# from app.models.paper import Paper
# from ..db import db
# from .route_utilities import validate_model, create_model

# bp = Blueprint("bp_paper", __name__, url_prefix="/papers")

# @bp.put("/<paper_id>")
# def update_paper_likes(paper_id):
#     paper = validate_model(Paper, paper_id)
#     paper.likes_count += 1
#     db.session.commit()
#     return {"paper": paper.to_dict()}, 200

# @bp.delete("/<paper_id>")
# def delete_paper(paper_id):
#     paper = validate_model(Paper, paper_id)
#     db.session.delete(paper)
#     db.session.commit()
#     return Response(status=204, mimetype="application/json")

# @bp.patch("/<paper_id>")
# def update_paper_fields(paper_id):
#     paper = validate_model(Paper, paper_id)
#     data = request.get_json()

#     if "title" in data:
#         paper.title = data["title"]
#     if "abstract" in data:
#         paper.abstract = data["abstract"]
#     if "authors" in data:
#         paper.authors = data["authors"]
#     if "publicationDate" in data:
#         paper.publicationDate = data["publicationDate"]
#     if "source" in data:
#         paper.source = data["source"]
#     if "URL" in data:
#         paper.URL = data["URL"]

#     db.session.commit()
#     return {"paper": paper.to_dict()}, 200

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