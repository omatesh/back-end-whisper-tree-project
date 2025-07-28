from sqlalchemy.orm import Mapped, mapped_column, relationship
from ..db import db
from sqlalchemy import ForeignKey
from typing import TYPE_CHECKING, Optional # Optional for CORE ID
from datetime import date
from datetime import datetime

# avoids import errors for Collection
if TYPE_CHECKING:
    from .collection import Collection

class Paper(db.Model):
    paper_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str]
    abstract: Mapped[str]
    authors: Mapped[str]
    publication_date: Mapped[Optional[date]] = mapped_column(nullable=True) # sOptional for nullable date
    source: Mapped[str]
    url: Mapped[str]
    likes_count: Mapped[int] = mapped_column(default=0)
    core_id: Mapped[Optional[str]] = mapped_column(nullable=True, unique=True) # field for CORE ID

    collection_id: Mapped[int] = mapped_column(ForeignKey("collection.collection_id"))
    collection: Mapped["Collection"] = relationship(back_populates="papers")

    def to_dict(self):
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "publication_date": (
                self.publication_date.isoformat() if self.publication_date else None
            ),
            "source": self.source,
            "url": self.url,
            "likes_count": self.likes_count,
            "collection_id": self.collection_id,
            "core_id": self.core_id # field for CORE ID
        }

    @classmethod
    def from_dict(cls, paper_data):
        pub_date_str = paper_data.get("publication_date")
        pub_date = None
        if pub_date_str:
            try:
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
            except ValueError:
                pub_date = None

        return Paper(
            title=paper_data["title"],
            abstract=paper_data.get("abstract", ""),
            authors=paper_data.get("authors", ""),
            publication_date=pub_date,
            source=paper_data["source"],
            url=paper_data.get("URL") or paper_data.get("url") or "",
            likes_count=paper_data.get("likes_count", 0),
            collection_id=paper_data.get("collection_id", None),
            core_id=paper_data.get("core_id", None) # field for CORE ID
        )
