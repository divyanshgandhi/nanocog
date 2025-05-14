"""
Retrieval-Aware Prompt Composer for Nano-Cog using ChromaDB
"""

import os
import chromadb
from chromadb import errors as chromadb_errors
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config import load_config


class RetrievalSystem:
    """
    Handles document retrieval and prompt composition with retrieved content
    """

    def __init__(self, config_path=None):
        """
        Initialize retrieval system

        Args:
            config_path (str, optional): Path to config file
        """
        self.config = load_config(config_path)
        self.db_path = self.config["retrieval"]["db_path"]
        self.embedding_model_name = self.config["retrieval"]["embedding_model"]
        self.num_results = self.config["retrieval"]["num_results"]
        self.similarity_threshold = self.config["retrieval"]["similarity_threshold"]

        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Initialize ChromaDB
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB client and collection"""
        # Create directory for DB if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize client with persistent storage
        self.client = chromadb.PersistentClient(
            path=os.path.dirname(self.db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        try:
            self.collection = self.client.get_collection("nano_cog_docs")
            print(f"Found existing collection with {self.collection.count()} documents")
        except (ValueError, chromadb_errors.NotFoundError):
            print("Creating new document collection")
            self.collection = self.client.create_collection(
                name="nano_cog_docs",
                metadata={"description": "Nano-Cog knowledge base"},
            )

            # Seed the collection with some initial data
            print("Seeding the collection with initial data")
            self.seed_from_wikipedia(10)
            self.seed_from_code_snippets(5)

    def add_documents(self, documents, metadatas=None, ids=None):
        """
        Add documents to the retrieval database

        Args:
            documents (list): List of text documents to add
            metadatas (list, optional): List of metadata dicts for each document
            ids (list, optional): List of IDs for each document
        """
        if ids is None:
            # Generate IDs if not provided
            ids = [f"doc_{i}_{hash(doc) % 10000}" for i, doc in enumerate(documents)]

        if metadatas is None:
            # Create empty metadata if not provided
            metadatas = [{"source": "unknown"} for _ in documents]

        # Generate embeddings and add to collection
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

        print(f"Added {len(documents)} documents to retrieval database")

    def retrieve(self, query, num_results=None):
        """
        Retrieve relevant documents for a query

        Args:
            query (str): Query text
            num_results (int, optional): Number of results to return

        Returns:
            list: List of retrieved documents with metadata
        """
        if num_results is None:
            num_results = self.num_results

        # Query the collection
        results = self.collection.query(query_texts=[query], n_results=num_results)

        # Process results
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else None

        retrieved_docs = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            # Skip documents below similarity threshold
            if distances and (1 - distances[i]) < self.similarity_threshold:
                continue

            retrieved_docs.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "similarity": 1 - distances[i] if distances else None,
                }
            )

        return retrieved_docs

    def compose_prompt(self, query, system_prompt=None):
        """
        Compose a prompt with retrieved documents

        Args:
            query (str): User query
            system_prompt (str, optional): System prompt to prepend

        Returns:
            str: Composed prompt with retrieved docs
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)

        # Compose prompt
        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Use the following documents to answer the question."

        docs_text = ""
        for i, doc in enumerate(retrieved_docs):
            # Sanitize the document text to fix PIPE-01
            doc_text = doc["text"]
            # Encode and decode to handle encoding issues, limit to 1024 chars, remove HTML
            doc_text = doc_text.encode("utf-8", "ignore").decode("utf-8")
            doc_text = re.sub(r"<[^>]+>", "", doc_text)[:1024]
            docs_text += f"\nSOURCE {i+1}: {doc_text}"

        # Format final prompt
        if docs_text:
            composed_prompt = f"SYSTEM{{{system_prompt}\n\nRELEVANT DOCUMENTS: {docs_text}}}\n\nUSER: {query}"
        else:
            composed_prompt = f"SYSTEM{{{system_prompt}}}\n\nUSER: {query}"

        return composed_prompt

    def seed_from_wikipedia(self, num_docs=50000):
        """
        Seed the database with Wikipedia sentences (placeholder)

        Args:
            num_docs (int): Number of Wikipedia documents to load

        Note: In a real implementation, this would download and process Wikipedia data
        """
        print(f"Seeding database with {num_docs} Wikipedia sentences (placeholder)")
        # This is a placeholder - actual implementation would use Wikipedia API or dumps

        # For demonstration, add a few sample documents
        sample_docs = [
            "Mamba is a state space model architecture for efficient sequence modeling.",
            "Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning method.",
            "Mixture of Experts (MoE) are sparse activation neural networks.",
            "Chain-of-thought prompting improves reasoning in language models.",
        ]

        sample_metadata = [
            {"source": "wikipedia", "category": "machine_learning"},
            {"source": "wikipedia", "category": "fine_tuning"},
            {"source": "wikipedia", "category": "neural_networks"},
            {"source": "wikipedia", "category": "prompting"},
        ]

        self.add_documents(sample_docs, sample_metadata)

    def seed_from_code_snippets(self, num_snippets=5000):
        """
        Seed the database with code snippets (placeholder)

        Args:
            num_snippets (int): Number of code snippets to load

        Note: In a real implementation, this would load code from repositories
        """
        print(f"Seeding database with {num_snippets} code snippets (placeholder)")
        # This is a placeholder - actual implementation would use code repositories

        # For demonstration, add a few sample code snippets
        sample_snippets = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "import torch\nx = torch.randn(3, 4)\ny = torch.nn.Linear(4, 2)(x)",
            "from transformers import AutoModel\nmodel = AutoModel.from_pretrained('bert-base-uncased')",
        ]

        sample_metadata = [
            {"source": "code", "language": "python", "category": "algorithm"},
            {"source": "code", "language": "python", "category": "pytorch"},
            {"source": "code", "language": "python", "category": "transformers"},
        ]

        self.add_documents(sample_snippets, sample_metadata)


if __name__ == "__main__":
    # Test retrieval system
    retrieval = RetrievalSystem()

    # Seed with sample data if collection is empty
    if retrieval.collection.count() == 0:
        print("Seeding retrieval database with sample data")
        retrieval.seed_from_wikipedia(10)
        retrieval.seed_from_code_snippets(5)

    # Test retrieval
    query = "How do state space models work?"
    retrieved = retrieval.retrieve(query)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved)} documents:")
    for i, doc in enumerate(retrieved):
        print(f"{i+1}. {doc['text']} (similarity: {doc['similarity']:.3f})")

    # Test prompt composition
    composed = retrieval.compose_prompt(query)
    print("\nComposed Prompt:")
    print(composed)
