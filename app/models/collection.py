from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..db import db

# avoids import errors for Paper
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .paper import Paper

class Collection(db.Model):

    collection_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str]
    owner: Mapped[str]
    papers: Mapped[list["Paper"]] = relationship(back_populates="collection")

    def to_dict(self):
        collection_as_dict = {
            "collection_id": self.collection_id,
            "title": self.title,
            "owner": self.owner 
        }
        return collection_as_dict
    
    @classmethod
    def from_dict(cls, collection_data):
        new_collection = Collection(
            title = collection_data["title"],
            owner = collection_data["owner"]
        )
        return new_collection