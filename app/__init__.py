from flask import Flask
from flask_cors import CORS
import os

# Import models, blueprints, and anything else needed to set up the app or database
from .db import db, migrate
from .models import collection, paper
from .routes.paper_routes import bp as paper_bp
from .routes.collection_routes import bp as collection_bp
from dotenv import load_dotenv

load_dotenv()

def create_app(config=None):
    app = Flask(__name__)

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')

    # For CORS =>
    app.config['CORS_HEADERS'] = 'Content-Type'

    if config:
        app.config.update(config)

    # Initialize app with SQLAlchemy db and Migrate
    db.init_app(app)
    migrate.init_app(app, db)

    # Register Blueprints 
    app.register_blueprint(collection_bp)
    app.register_blueprint(paper_bp)

    CORS(app)
    return app