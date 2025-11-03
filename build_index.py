from src.faiss_index import FAISSIndex
from src.embeddings import EmbeddingModel
from src.config import DOCUMENTS_JSONL_PATH


def main():
    """Main entry point for index building."""
    print("=" * 80)
    print("FAISS INDEX BUILDER")
    print("=" * 80)
    print()
    
    embedding_model = EmbeddingModel()
    faiss_index = FAISSIndex()
    
    print("Building index...")
    print("-" * 80)
    faiss_index.build(
        documents_path=DOCUMENTS_JSONL_PATH,
        embedding_model=embedding_model
    )
    
    print("\nSaving index...")
    print("-" * 80)
    faiss_index.save()
    print("\n" + "=" * 80)
    print("FAISS INDEX BUILD COMPLETE!")


if __name__ == "__main__":
    main()

