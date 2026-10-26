"""
memory.py — Long-term memory backed by Chroma (local vector database).

How it works, in simpler terms:
  1. When the agent saves a note, Chroma turns the text into an "embedding"
     (a list of numbers that captures its meaning) and stores it on disk.
  2. When the agent searches, the query is embedded too, and Chroma returns
     the notes whose numbers are closest — i.e. most similar in MEANING,
     not just matching keywords. ("my pet's name" can find "my dog is Max".)

Chroma's default embedding model (all-MiniLM-L6-v2) runs locally on your CPU,
so there's no API key and no cost!
"""

import uuid
import hashlib
from datetime import datetime
import chromadb
from ddgs.cli import text

class Memory:
    def __init__(self, path: str = "./chroma_db", embedding_function=None,
                 collection_name: str = "notes"):
        # PersistentClient saves to a folder, so memory survive the restart.
        self.client = chromadb.PersistentClient(path=path)

        # A "collection" is like a table. Passing embedding_function = None
        # tells Chroma to use its free built-in local model.
        kwargs = {"name": collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self.collection = self.client.get_or_create_collection(**kwargs)

    def add(self, text: str) -> str:
        """
        Store one note and return its id.
        Saving the exact same text twice won't duplicate it.
        """
        # Same text -> same id, so upsert overwrites instead of duplicating.
        note_id = hashlib.md5(text.strip().lower().encode()).hexdigest()
        self.collection.upsert(
            ids=[note_id],
            documents=[text],
            # Metadata is extra info stored alongside the text.
            metadatas=[{"saved_at": datetime.now().isoformat(timespec="seconds")}],
        )
        return note_id

    def search(self, query: str, k: int = 3,
               max_distance: float | None = None) -> list[str]:
        """
        Return up to k notes most similar in meaning to the query.

        max_distance drops weak matches. Chroma's default distance is squared
        L2; on MiniLM's normalized vectors that's 2 - 2*cosine, so 0 is
        identical and ~1.0 is cosine 0.5. None keeps the top k regardless.
        """
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, count),
                                        include=["documents", "distances"])
        # Chroma returns a list per query; we sent one query, so take [0].
        docs, distances = results["documents"][0], results["distances"][0]
        if max_distance is None:
            return docs
        return [d for d, dist in zip(docs, distances) if dist <= max_distance]

    def count(self) -> int:
        return self.collection.count()
