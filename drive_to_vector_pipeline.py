import os
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import json
from typing import List, Dict, Optional
import tempfile
from dotenv import load_dotenv

# Pinecone and embedding imports
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import uuid
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

SCOPES = [
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/drive.metadata.readonly',
    'https://www.googleapis.com/auth/drive'
]

class DriveToVectorPipeline:
    def __init__(self, credentials_file='credentials.json', token_file='token.json', 
                 pinecone_api_key=None, index_name='drive-docs-rag', search_only=False):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.index_name = index_name
        self.batch_size = 100  # Upsert in batches of 100
        self.pending_vectors = []  # Buffer for batching
        self.search_only = search_only
        
        # Initialize embedding model
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Initialize Pinecone
        if not pinecone_api_key:
            raise ValueError("Please provide your Pinecone API key")
        
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.setup_pinecone_index()
        
        # Authenticate with Google Drive
        if not self.search_only:
            self.authenticate()
    
    def authenticate(self):
        """Authenticate and create the Drive service"""
        creds = None
        
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())

        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive service authenticated successfully")
        
    def setup_pinecone_index(self):
        """Create or connect to Pinecone index"""
        print(f"🔄 Setting up Pinecone index: {self.index_name}")
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"📝 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Change to your preferred region
                )
            )
        else:
            print(f"✅ Using existing index: {self.index_name}")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        print(f"📊 Index stats: {self.index.describe_index_stats()}")
    
    def flush_batch(self):
        """Upsert any pending vectors to Pinecone"""
        if not self.pending_vectors:
            return
        
        print(f"📤 Upserting batch of {len(self.pending_vectors)} vectors...")
        
        try:
            self.index.upsert(vectors=self.pending_vectors)
            print(f"✅ Successfully upserted {len(self.pending_vectors)} vectors")
            self.pending_vectors = []
        except Exception as e:
            print(f"❌ Error upserting batch: {str(e)}")
            # Don't clear the batch on error - could retry later
    
    def add_to_batch(self, vector_id: str, embedding: list, metadata: dict):
        """Add vector to batch and flush if batch is full"""
        self.pending_vectors.append({
            'id': vector_id,
            'values': embedding,
            'metadata': metadata
        })
        
        if len(self.pending_vectors) >= self.batch_size:
            self.flush_batch()
    
    def get_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """Get file content as text from Google Drive API"""
        try:
            # Handle Google Docs, Sheets, Slides
            if mime_type.startswith('application/vnd.google-apps'):
                export_formats = {
                    'application/vnd.google-apps.document': 'text/plain',
                    'application/vnd.google-apps.spreadsheet': 'text/csv',
                    'application/vnd.google-apps.presentation': 'text/plain'
                }
                
                if mime_type in export_formats:
                    request = self.service.files().export_media(
                        fileId=file_id, 
                        mimeType=export_formats[mime_type]
                    )
                else:
                    print(f"⚠️  Unsupported Google Apps file type: {mime_type}")
                    return None
            else:
                # Regular file download
                request = self.service.files().get_media(fileId=file_id)
            
            # Download to memory
            file_io = io.BytesIO()
            downloader = MediaIoBaseDownload(file_io, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Convert to text based on file type
            content = file_io.getvalue()
            
            if mime_type.startswith('text/') or 'plain' in mime_type:
                return content.decode('utf-8', errors='ignore')
            elif mime_type == 'application/pdf':
                return self.extract_pdf_text(content)
            elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return self.extract_docx_text(content)
            elif 'csv' in mime_type or 'spreadsheet' in mime_type:
                return content.decode('utf-8', errors='ignore')
            else:
                # Try to decode as text anyway
                try:
                    return content.decode('utf-8', errors='ignore')
                except:
                    print(f"⚠️  Could not extract text from {mime_type}")
                    return None
                    
        except Exception as e:
            print(f"❌ Error reading file content: {str(e)}")
            return None
    
    def extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes (requires PyPDF2 or similar)"""
        try:
            import PyPDF2
            with io.BytesIO(pdf_content) as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            print("⚠️  PyPDF2 not installed, skipping PDF extraction")
            return None
        except Exception as e:
            print(f"⚠️  Error extracting PDF: {str(e)}")
            return None
    
    def extract_docx_text(self, docx_content: bytes) -> str:
        """Extract text from DOCX bytes (requires python-docx)"""
        try:
            from docx import Document
            with io.BytesIO(docx_content) as docx_file:
                doc = Document(docx_file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
        except ImportError:
            print("⚠️  python-docx not installed, skipping DOCX extraction")
            return None
        except Exception as e:
            print(f"⚠️  Error extracting DOCX: {str(e)}")
            return None
    
    def get_folder_path(self, file_item: Dict, folder_cache: Dict = None) -> str:
        """Get the full folder path for a file"""
        if folder_cache is None:
            folder_cache = {}
        
        def get_parent_path(parent_id):
            if parent_id in folder_cache:
                return folder_cache[parent_id]
            
            try:
                parent = self.service.files().get(
                    fileId=parent_id,
                    fields="id, name, parents",
                    supportsAllDrives=True  # Add this for shared drives
                ).execute()
                
                if 'parents' in parent:
                    grandparent_path = get_parent_path(parent['parents'][0])
                    path = f"{grandparent_path}/{parent['name']}" if grandparent_path else parent['name']
                else:
                    path = parent['name']
                
                folder_cache[parent_id] = path
                return path
            except:
                return ""
        
        if 'parents' in file_item:
            return get_parent_path(file_item['parents'][0])
        return ""
    
    def create_document_metadata(self, file_item: Dict, folder_path: str) -> Dict:
        """Create metadata for vector database"""
        return {
            'file_id': file_item['id'],
            'name': file_item['name'],
            'folder_path': folder_path,
            'full_path': f"{folder_path}/{file_item['name']}" if folder_path else file_item['name'],
            'mime_type': file_item['mimeType'],
            'size': file_item.get('size'),
            'modified_time': file_item.get('modifiedTime'),
            'source': 'google_drive'
        }
    
    def simple_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Simple text chunking function"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence ending
                for punct in ['. ', '\n\n', '\n', '. ']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + chunk_size // 2:
                        end = last_punct + len(punct)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def process_and_embed_file(self, file_item: Dict, folder_path: str):
        """Process a single file and add to vector database"""
        print(f"📄 Processing: {folder_path}/{file_item['name']}")
        
        # Get file content
        content = self.get_file_content(file_item['id'], file_item['mimeType'])
        if not content or len(content.strip()) == 0:
            print(f"⚠️  No text content found, skipping")
            return
        
        # Create base metadata
        base_metadata = self.create_document_metadata(file_item, folder_path)
        
        # Split text into chunks
        chunks = self.simple_chunk_text(content, chunk_size=1000, overlap=200)
        print(f"📊 Created {len(chunks)} chunks from {file_item['name']}")
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Create unique ID for this chunk
                vector_id = f"{file_item['id']}_chunk_{i}"
                
                # Create chunk metadata (Pinecone has metadata size limits)
                chunk_metadata = {
                    'file_id': base_metadata['file_id'],
                    'file_name': base_metadata['name'],
                    'folder_path': base_metadata['folder_path'],
                    'full_path': base_metadata['full_path'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'mime_type': base_metadata['mime_type'],
                    'modified_time': base_metadata['modified_time'],
                    'chunk_text': chunk,  # Store full chunk
                    'chunk_length': len(chunk),
                    'source': 'google_drive',
                    'processed_at': datetime.now().isoformat()
                }
                
                # Add to batch
                self.add_to_batch(vector_id, embedding, chunk_metadata)
                
            except Exception as e:
                print(f"❌ Error processing chunk {i} of {file_item['name']}: {str(e)}")
                continue
    
    def process_all_files(self, folder_id: Optional[str] = None, shared_drive_id: Optional[str] = None):
        """Process all files and add to vector database"""
        print("🔍 Scanning Google Drive...")
        
        # Build query for shared drives
        if shared_drive_id:
            query = f"'{folder_id}' in parents" if folder_id else ""
            
            # Get all files from shared drive
            results = self.service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, parents, size, modifiedTime)",
                supportsAllDrives=True,  # Key for shared drives
                includeItemsFromAllDrives=True,  # Key for shared drives
                driveId=shared_drive_id,  # Specify which shared drive
                corpora='drive'  # Search within specific drive
            ).execute()
        else:
            # Original code for personal drive
            query = f"'{folder_id}' in parents" if folder_id else ""
            results = self.service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, parents, size, modifiedTime)"
            ).execute()
        
        files = results.get('files', [])
        print(f"📁 Found {len(files)} items")
        
        # Rest of the method stays the same...
        folder_cache = {}
        processed = 0
        
        for file_item in files:
            # Skip folders
            if file_item['mimeType'] == 'application/vnd.google-apps.folder':
                continue
            
            # Skip very large files (adjust threshold as needed)
            if file_item.get('size') and int(file_item['size']) > 50 * 1024 * 1024:  # 50MB
                print(f"⚠️  Skipping large file: {file_item['name']} ({int(file_item['size'])/(1024*1024):.1f}MB)")
                continue
            
            folder_path = self.get_folder_path(file_item, folder_cache)
            
            try:
                self.process_and_embed_file(file_item, folder_path)
                processed += 1
            except Exception as e:
                print(f"❌ Error processing {file_item['name']}: {str(e)}")
                continue
        
        print(f"✅ Processed {processed} files")
        self.flush_batch()
        print(f"📊 Final index stats: {self.index.describe_index_stats()}")

    def search_documents(self, query: str, top_k: int = 10, filter_dict: dict = None):
        """Search for documents using the query"""
        print(f"🔍 Searching for: '{query}'")
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict  # e.g., {"folder_path": {"$eq": "Marketing"}}
        )
        
        print(f"📋 Found {len(results['matches'])} results")
        
        for i, match in enumerate(results['matches']):
            print(f"\n{i+1}. Score: {match['score']:.3f}")
            print(f"   File: {match['metadata']['full_path']}")
            print(f"   Chunk: {match['metadata']['chunk_id']}/{match['metadata']['total_chunks']}")
            print(f"   Preview: {match['metadata']['chunk_text'][:200]}...")
        
        return results
        
    def list_shared_drives(self):
        """List all shared drives the user has access to"""
        print("📋 Listing shared drives...")
        
        results = self.service.drives().list(pageSize=100).execute()
        drives = results.get('drives', [])
        
        print(f"Found {len(drives)} shared drives:")
        for i, drive in enumerate(drives):
            print(f"  {i+1}. {drive['name']} (ID: {drive['id']})")
        
        return drives

    def find_folder_in_shared_drive(self, shared_drive_id: str, folder_name: str):
        """Find a folder by name in a shared drive"""
        print(f"🔍 Looking for folder '{folder_name}' in shared drive...")
        
        results = self.service.files().list(
            q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            driveId=shared_drive_id,
            corpora='drive',
            fields="files(id, name, parents)"
        ).execute()
        
        folders = results.get('files', [])
        
        if folders:
            folder = folders[0]  # Take first match
            print(f"✅ Found folder: {folder['name']} (ID: {folder['id']})")
            return folder['id']
        else:
            print(f"❌ Folder '{folder_name}' not found")
            return None

    def process_recent_files(self, shared_drive_id: str, folder_id: Optional[str] = None, days_back: int = 7):
        """Process only files modified in the last N days"""
        from datetime import datetime, timedelta
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_iso = cutoff_date.isoformat() + 'Z'
        
        print(f"🔍 Scanning for files modified after {cutoff_date.strftime('%Y-%m-%d %H:%M')}...")
        
        # Build query with date filter
        date_filter = f"modifiedTime > '{cutoff_iso}'"
        
        if folder_id:
            query = f"'{folder_id}' in parents and {date_filter}"
        else:
            query = date_filter
        
        # Get files from shared drive with date filter
        results = self.service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType, parents, size, modifiedTime)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            driveId=shared_drive_id,
            corpora='drive',
            orderBy='modifiedTime desc'  # Most recent first
        ).execute()
        
        files = results.get('files', [])
        print(f"📁 Found {len(files)} files modified in last {days_back} days")
        
        if not files:
            print("ℹ️  No recent files found. Try increasing days_back parameter.")
            return 0
        
        # Show what we found
        print("\n📋 Recent files to process:")
        for i, file_item in enumerate(files[:10]):  # Show first 10
            modified = file_item.get('modifiedTime', 'Unknown')
            size_mb = int(file_item.get('size', 0)) / (1024*1024) if file_item.get('size') else 0
            print(f"  {i+1}. {file_item['name']} ({size_mb:.1f}MB) - {modified}")
        
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        
        # Process files
        folder_cache = {}
        processed = 0
        
        for file_item in files:
            # Skip folders
            if file_item['mimeType'] == 'application/vnd.google-apps.folder':
                continue
            
            # Skip very large files
            if file_item.get('size') and int(file_item['size']) > 20 * 1024 * 1024:  # 20MB limit for demo
                print(f"⚠️  Skipping large file: {file_item['name']} ({int(file_item['size'])/(1024*1024):.1f}MB)")
                continue
            
            folder_path = self.get_folder_path(file_item, folder_cache)
            
            try:
                self.process_and_embed_file(file_item, folder_path)
                processed += 1
                print(f"✅ Processed {processed}/{len([f for f in files if f['mimeType'] != 'application/vnd.google-apps.folder'])}")
            except Exception as e:
                print(f"❌ Error processing {file_item['name']}: {str(e)}")
                continue
        
        print(f"\n🎉 Demo processing complete!")
        print(f"   Processed: {processed} files")
        self.flush_batch()
        
        # Show final stats
        stats = self.index.describe_index_stats()
        print(f"📊 Pinecone index now contains: {stats['total_vector_count']} vectors")
        
        return processed


# Usage example
if __name__ == "__main__":
    # Initialize pipeline with your Pinecone API key
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    
    pipeline = DriveToVectorPipeline(
        pinecone_api_key=PINECONE_API_KEY,
        index_name="drive-docs-rag"  # Your index name
    )

    shared_drives = pipeline.list_shared_drives()
    print(shared_drives)
    
    # Process all accessible files
    # pipeline.process_all_files()
    
    # Example search after processing
    # pipeline.search_documents("frederick douglass quote about light and darkness reconstruction", top_k=5)

    
    # Search with folder filter
    # pipeline.search_documents("budget", filter_dict={"folder_path": {"$eq": "Marketing/Campaigns"}})