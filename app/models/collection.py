from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional
from ..db import db

# avoids import errors for Paper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .paper import Paper

class Collection(db.Model):

    collection_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str]
    owner: Mapped[Optional[str]] = mapped_column(nullable=True)
    description: Mapped[str]
    papers: Mapped[list["Paper"]] = relationship(back_populates="collection")

    def to_dict(self):
        collection_as_dict = {
            "collection_id": self.collection_id,
            "title": self.title,
            "owner": self.owner,
            "description": self.description 
        }
        return collection_as_dict
    # def to_dict(self, include_papers=False):
    #     collection_as_dict = {
    #         "collection_id": self.collection_id,
    #         "title": self.title,
    #         "owner": self.owner,
    #         "description": self.description 
    #     }
        
    #     if include_papers:
    #         collection_as_dict["papers"] = [paper.to_dict() for paper in self.papers]
    #     else:
    #         collection_as_dict["papers_count"] = len(self.papers)
        
    #     return collection_as_dict
    
    @classmethod
    def from_dict(cls, collection_data):
        new_collection = Collection(
            title = collection_data["title"],
            owner = collection_data["owner"],
            description = collection_data["description"],
        )
        return new_collection