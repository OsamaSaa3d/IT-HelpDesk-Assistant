from src.data_processing import TicketDataProcessor
from src.config import ensure_directories


def main():

    print("=" * 80)
    print("TICKET DATA PREPARATION")
    print("=" * 80)
    print()
    
    # Ensure directories exist
    ensure_directories()
    
    processor = TicketDataProcessor()
    
    print("Step 1: Unifying tickets from multiple sources")
    print("-" * 80)
    df_unified = processor.unify_tickets()
    
    print("\nStep 2: Creating document objects")
    print("-" * 80)
    documents = processor.create_documents(df_unified)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed {len(documents)} tickets")


if __name__ == "__main__":
    main()

