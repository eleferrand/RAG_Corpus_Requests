
import json
import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def load_corpus(json_path, language):
    """Load aligned sentences and glosses from JSON."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for sentence_id, entry in data.items():

        text = entry.get("text", "").strip()
        gloss = entry.get("gloss", "").strip()

        if not text or not gloss:
            print(f"Skipping incomplete entry: {sentence_id}")
            continue

        # One document per sentence.
        # Embed the gloss; preserve the original sentence.
        documents.append(
            Document(
                page_content=gloss,
                metadata={
                    "sentence_id": str(sentence_id),
                    "language": language,
                    "text": text,
                    "gloss": gloss,
                    "source": Path(json_path).name
                }
            )
        )

    return documents


def create_vectordb(
    documents,
    persist_directory,
    collection_name,
    embedding_model
):
    embeddings = OllamaEmbeddings(model=embedding_model)

    # Remove the previous collection if it exists
    old_db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    old_db.delete_collection()

    # Use stable IDs to identify individual sentences
    ids = [
        f"{doc.metadata['language']}_{doc.metadata['sentence_id']}"
        for doc in documents
    ]

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    return vectordb

def main():

    parser = argparse.ArgumentParser(
        description="Ingest an interlinear glossed corpus into ChromaDB."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the JSON corpus"
    )

    parser.add_argument(
        "--language",
        default="Amis",
        help="Language name"
    )

    parser.add_argument(
        "--db",
        default="./chroma_db",
        help="ChromaDB persistence directory"
    )

    parser.add_argument(
        "--collection",
        default="amis_glosses",
        help="ChromaDB collection name"
    )

    parser.add_argument(
        "--model",
        default="bge-m3",
        help="Ollama embedding model"
    )

    args = parser.parse_args()

    print("Loading corpus...")

    documents = load_corpus(
        args.input,
        args.language
    )

    print(f"Loaded {len(documents)} sentences.")

    print("Creating embeddings and vector database...")

    vectordb = create_vectordb(
        documents=documents,
        persist_directory=args.db,
        collection_name=args.collection,
        embedding_model=args.model
    )

    print("\nIngestion complete!")
    print(f"Documents indexed: {vectordb._collection.count()}")
    print(f"Database location: {args.db}")
    print(f"Collection: {args.collection}")


if __name__ == "__main__":
    main()
