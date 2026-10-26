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
from datetime import datetime
import chromadb

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
        """
        note_id = str(uuid.uuid4())
        self.collection.add(
            ids=[note_id],
            documents=[text],
            # Metadata is extra info stored alongside the text.
            metadatas=[{"saved_at": datetime.now().isoformat(timespec="seconds")}],
        )
        return note_id

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        Return up to k notes most similar in meaning to the query.
        """
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(query_texts=[query], n_results=min(k, count))
        # Chroma returns a list per query; we sent one query, so take [0].
        return results["documents"][0]

    def count(self) -> int:
        return self.collection.count()
