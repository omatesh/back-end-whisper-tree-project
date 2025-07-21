from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..db import db
from sqlalchemy import ForeignKey

# avoids import errors for Collection
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .collection import Collection

class Paper(db.Model):
    paper_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    message: Mapped[str]
    likes_count: Mapped[int] = mapped_column(default=0)

    collection_id: Mapped[int] = mapped_column(ForeignKey("collection.collection_id")) 
    collection: Mapped["Collection"] = relationship(back_populates="papers")
    
    def to_dict(self):
        paper_as_dict = {
            "paper_id": self.paper_id,
            "message": self.message,
            "likes_count": self.likes_count,
            "collection_id": self.collection_id
        }

        return paper_as_dict
    
    @classmethod
    def from_dict(cls, paper_data):
        new_paper = Paper(
            message = paper_data["message"],
            likes_count=paper_data.get("likes_count", 0),
            collection_id = paper_data["collection_id"]
        )
        return new_paper