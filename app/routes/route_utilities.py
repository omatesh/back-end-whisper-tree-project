from flask import abort, make_response
import requests
from ..db import db
import os

def validate_model(cls, model_id):
    try:
        model_id = int(model_id)
    except:
        response = {"details": "Invalid data"}
        abort(make_response(response , 400))

    # line below used when the model's primary key column is literally named id
    # query = db.select(cls).where(cls.id == model_id)

    #code below is used when the model’s primary key column has a unique
    #it returns a SQLAlchemy Column object representing the primary key column of a model

    primary_key_column = next(iter(cls.__mapper__.primary_key))
    query = db.select(cls).where(primary_key_column == model_id)

    model = db.session.scalar(query)
    
    if not model:
        response = {"message": f"{cls.__name__} {model_id} not found"}
        abort(make_response(response, 404))
    
    return model


def create_model(cls, model_data):
    try:
        new_model = cls.from_dict(model_data)
        
    except KeyError as error:
        response = {"details": "Invalid data"}
        abort(make_response(response, 400))
    
    db.session.add(new_model)
    db.session.commit()

    return {f"{cls.__name__.lower()}": new_model.to_dict()}, 201
