# Document Search Tool

A dual-mode application for semantic search over your Google Drive documents:

- **Desktop GUI**: PyQt5-based interactive application
- **Web App**: Streamlit-based browser interface

Under the hood, it uses Sentence-Transformers to embed document chunks, stores vectors in Pinecone, and offers both local and web-driven search experiences.

## Features

- **Google Drive ingestion**: fetch Docs, Sheets, Slides, PDFs, Word, CSV, and text files
- **Chunking & embedding**: splits text into ~1k-token chunks (200 token overlap) and encodes with `all-MiniLM-L6-v2`
- **Vector storage & search**: batch-upserts to a Pinecone index and queries with cosine similarity
- **Desktop GUI**: PyQt5 app with collapsible result groups, background search thread, and auto-update checks
- **Web App**: Streamlit interface with live search, adjustable parameters, and result previews
- **Auto-updater**: checks GitHub releases and downloads new versions seamlessly (desktop only)

## Repository Structure

- `drive_to_vector_pipeline.py` – Core pipeline: authenticate Drive API, extract & chunk content, embed, and upsert to Pinecone.
- `search_gui.py` – PyQt5 desktop application.
- `streamlit_app.py` – Streamlit web application.
- `updater.py` – `AutoUpdater` and `UpdateChecker` classes for GitHub-based updates (desktop).
- `requirements-desktop.txt` – Dependencies for the desktop GUI.
- `requirements.txt` – Dependencies for the web app.
- `README.md` – This file.

## Setup

### Prerequisites

- Python 3.8+
- Google Cloud project with **Drive API** enabled
- Pinecone account & API key
- (Optional, desktop) GitHub repo for update checks

### Installation

```bash
git clone https://github.com/your-username/document-search-tool.git
cd document-search-tool
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

#### Desktop GUI

```bash
pip install -r requirements-desktop.txt
```

#### Web App

```bash
pip install -r requirements.txt
```

### Google Drive API

1. Enable the **Google Drive API** in Google Cloud Console.
2. Create **OAuth 2.0 Client Credentials** (Desktop App).
3. Download `credentials.json` into the project root.
4. On first run, the app will launch a browser prompt and save `token.json`.

### Pinecone

1. Sign up at [Pinecone](https://www.pinecone.io/) and create an API key.
2. Export your key:

   ```bash
   export PINECONE_API_KEY="YOUR_KEY"
   ```

   or add a `.env` file in the project root:

   ```
   PINECONE_API_KEY=YOUR_KEY
   ```

### (Optional) GitHub Updates (Desktop Only)

To enable the auto-updater:

```bash
export GITHUB_USERNAME=your-github-username
```

Ensure `updater.py`’s `self.github_repo` matches your repo path.

## Usage

### 1. Index Your Drive

```python
from drive_to_vector_pipeline import DriveToVectorPipeline
import os

pipeline = DriveToVectorPipeline(
  credentials_file='credentials.json',
  token_file='token.json',
  pinecone_api_key=os.getenv('PINECONE_API_KEY'),
  index_name='drive-docs-rag'      # customize as needed
)
pipeline.process_all_files()        # scans entire Drive
```

To index a specific folder:

```python
pipeline.process_all_files(folder_id='YOUR_FOLDER_ID')
```

### 2. Desktop GUI

```bash
python search_gui.py
```

- Enter a query and click **Search**
- Results grouped by document with top-scoring chunks
- Desktop app auto-checks for updates under **Help → Check for Updates**

### 3. Web App

```bash
streamlit run streamlit_app.py
```

- Open the provided local URL in your browser
- Enter queries, adjust `top_k` and other sidebar options
- View results and document previews inline

## Configuration

- **Chunk size & overlap**: adjust `simple_chunk_text(chunk_size, overlap)` in `drive_to_vector_pipeline.py`.
- **Index settings**: modify `dimension`, `metric`, or `ServerlessSpec` in `setup_pinecone_index()`.
- **Scopes**: change `SCOPES` if additional Drive permissions are needed.
- **GUI styling**: tweak fonts, layouts, and colors in `search_gui.py`.
- **Web parameters**: adjust sidebar options in `streamlit_app.py`.

## Contributing

Contributions welcome! Fork the repo, add tests, follow PEP8, and submit a pull request.

## License

This project is [MIT](LICENSE) licensed.
