# demo_setup.py
from drive_to_vector_pipeline import DriveToVectorPipeline
import os
from dotenv import load_dotenv

load_dotenv(override=True)
print(f"🔍 Debug: PINECONE_API_KEY = {os.getenv('PINECONE_API_KEY')[:10]}..." if os.getenv('PINECONE_API_KEY') else "❌ PINECONE_API_KEY not found")

def run_demo_setup():
    # Initialize pipeline
    print("🚀 Setting up Document Search Demo")
    
    pipeline = DriveToVectorPipeline(
        pinecone_api_key=os.getenv('PINECONE_API_KEY')
    )
    
    # Step 1: Find shared drives
    print("\n" + "="*50)
    shared_drives = pipeline.list_shared_drives()
    
    if not shared_drives:
        print("❌ No shared drives found")
        return
    
    # Step 2: Choose shared drive (or let user pick)
    shared_drive_id = shared_drives[0]['id']  # Use first one
    print(f"📂 Using shared drive: {shared_drives[0]['name']}")
    
    # Step 3: Process recent files (last 7 days)
    print("\n" + "="*50)
    processed = pipeline.process_recent_files(
        shared_drive_id=shared_drive_id, 
        days_back=7  # Last week
    )
    
    if processed > 0:
        print(f"\n🎉 Demo ready! {processed} recent files indexed.")
        print("💡 Now you can search with natural language queries!")
        
        # Test search
        print("\n" + "="*50)
        print("🔍 Testing search...")
        test_query = "meeting notes"
        results = pipeline.search_documents(test_query, top_k=3)
        
        if results['matches']:
            print(f"✅ Search working! Found {len(results['matches'])} results for '{test_query}'")
        else:
            print("ℹ️  No results for test query. Try a different search term.")
    else:
        print("⚠️  No files were processed. Try increasing days_back or check file types.")

if __name__ == "__main__":
    run_demo_setup()