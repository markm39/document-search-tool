import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLineEdit, QPushButton, 
                            QTextEdit, QLabel, QScrollArea, QFrame, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from drive_to_vector_pipeline import DriveToVectorPipeline
from collections import defaultdict

from updater import AutoUpdater

class CollapsibleGroupBox(QFrame):
    """A collapsible group box widget"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Box)
        self.setStyleSheet("QFrame { margin: 2px; }")
        
        self.layout = QVBoxLayout(self)
        
        # Header button
        self.toggle_button = QPushButton(f"▼ {title}")
        
        self.toggle_button.setStyleSheet("QPushButton { text-align: left; font-weight: bold; }")
        self.toggle_button.clicked.connect(self.toggle)
        self.layout.addWidget(self.toggle_button)
        
        # Content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.layout.addWidget(self.content_area)
        
        self.is_expanded = True
    
    def toggle(self):
        self.is_expanded = not self.is_expanded
        self.content_area.setVisible(self.is_expanded)
        arrow = "▼" if self.is_expanded else "▶"
        current_text = self.toggle_button.text()
        self.toggle_button.setText(f"{arrow}{current_text[1:]}")
    
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

class SearchWorker(QThread):
    """Background thread for searching to keep UI responsive"""
    results_ready = pyqtSignal(dict)  # Changed to dict for grouped results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, pipeline, query):
        super().__init__()
        self.pipeline = pipeline
        self.query = query
    
    def run(self):
        try:
            results = self.pipeline.search_documents(self.query, top_k=20)  # Get more chunks
            grouped_results = self.group_results_by_document(results['matches'])
            self.results_ready.emit(grouped_results)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def group_results_by_document(self, matches):
        """Group search results by document and rank documents by best match score"""
        document_groups = defaultdict(list)
        
        # Group matches by file_id
        for match in matches:
            file_id = match['metadata']['file_id']
            document_groups[file_id].append(match)
        
        # Sort chunks within each document by score and take top 3
        for file_id in document_groups:
            document_groups[file_id].sort(key=lambda x: x['score'], reverse=True)
            document_groups[file_id] = document_groups[file_id][:3]  # Top 3 chunks per doc
        
        # Sort documents by their best match score and take top 5
        sorted_docs = sorted(
            document_groups.items(),
            key=lambda x: x[1][0]['score'],  # Sort by best chunk score
            reverse=True
        )[:5]  # Top 5 documents
        
        return dict(sorted_docs)

class DocumentSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pipeline = None
        self.current_version = "1.0.0" #update with each release
        self.init_ui()
        self.init_pipeline()

        # Initialize auto-updater
        self.updater = AutoUpdater(self, self.current_version)
        
        # Check for updates on startup (silent)
        self.updater.check_for_updates(silent=True)
    
    def init_ui(self):
        self.setWindowTitle(f"Document Search Tool v{self.current_version}")
        self.setGeometry(100, 100, 900, 700)
        
        # Create menu bar first (before central widget on macOS)
        self.create_menu_bar()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Search Your Documents")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Search section
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query here...")
        self.search_input.setFont(QFont("Arial", 12))
        self.search_input.returnPressed.connect(self.search_documents)
        
        self.search_button = QPushButton("Search")
        self.search_button.setFont(QFont("Arial", 12))
        self.search_button.clicked.connect(self.search_documents)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        layout.addLayout(search_layout)
        
        # Add buttons for menu actions (more visible on all platforms)
        button_layout = QHBoxLayout()
        
        self.update_button = QPushButton("Check for Updates")
        self.update_button.clicked.connect(lambda: self.updater.check_for_updates(silent=False))
        
        self.about_button = QPushButton("About")
        self.about_button.clicked.connect(self.show_about)
        
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.about_button)
        button_layout.addStretch()  # Push buttons to the left
        layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Ready to search...")
        self.status_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.status_label)
        
        # Results area
        self.results_area = QScrollArea()
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_area.setWidget(self.results_widget)
        self.results_area.setWidgetResizable(True)
        layout.addWidget(self.results_area)

    def create_menu_bar(self):
        """Create menu bar (may not be visible on macOS due to system behavior)"""
        menubar = self.menuBar()
        
        # Force menu bar to be visible in the window on macOS
        menubar.setNativeMenuBar(False)
        
        help_menu = menubar.addMenu('Help')
        
        check_updates_action = help_menu.addAction('Check for Updates')
        check_updates_action.triggered.connect(lambda: self.updater.check_for_updates(silent=False))
        
        about_action = help_menu.addAction('About')
        about_action.triggered.connect(self.show_about)

    def show_about(self):
        QMessageBox.about(self, "About", 
            f"Document Search Tool\n"
            f"Version {self.current_version}\n\n"
            f"A semantic search tool for your Google Drive documents.\n"
            f"Built with PyQt5, Pinecone, and Sentence Transformers.")
    
    
    def init_pipeline(self):
        """Initialize the search pipeline"""
        try:
            self.status_label.setText("Loading search system...")
            
            # Load from environment or config file
            pinecone_api_key = os.environ.get('PINECONE_API_KEY')
            if not pinecone_api_key:
                # Try to load from a config file next to the executable
                config_path = os.path.join(os.path.dirname(sys.executable), 'config.txt')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        pinecone_api_key = f.read().strip()
            
            if not pinecone_api_key:
                self.status_label.setText("❌ No API key found. Please set PINECONE_API_KEY or create config.txt")
                return
            
            self.pipeline = DriveToVectorPipeline(
                pinecone_api_key=pinecone_api_key,
                index_name="drive-docs-rag"
            )
            self.status_label.setText("✅ Ready to search!")
            
        except Exception as e:
            self.status_label.setText(f"❌ Error: {str(e)}")
    
    def search_documents(self):
        if not self.pipeline:
            self.status_label.setText("❌ Search system not ready")
            return
        
        query = self.search_input.text().strip()
        if not query:
            return
        
        self.status_label.setText("🔍 Searching...")
        self.search_button.setEnabled(False)
        
        # Clear previous results
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Start search in background thread
        self.search_worker = SearchWorker(self.pipeline, query)
        self.search_worker.results_ready.connect(self.display_results)
        self.search_worker.error_occurred.connect(self.handle_error)
        self.search_worker.start()
    
    def display_results(self, grouped_results):
        self.search_button.setEnabled(True)
        
        if not grouped_results:
            self.status_label.setText("No results found")
            no_results = QLabel("No documents found matching your search.")
            no_results.setFont(QFont("Arial", 12))
            self.results_layout.addWidget(no_results)
            return
        
        total_docs = len(grouped_results)
        total_chunks = sum(len(chunks) for chunks in grouped_results.values())
        self.status_label.setText(f"Found {total_chunks} matches in {total_docs} documents")
        
        for doc_rank, (file_id, chunks) in enumerate(grouped_results.items(), 1):
            document_widget = self.create_document_widget(doc_rank, chunks)
            self.results_layout.addWidget(document_widget)
    
    def create_document_widget(self, doc_rank, chunks):
        """Create a collapsible widget for a document with its chunks"""
        # Get document info from the first chunk
        first_chunk = chunks[0]
        doc_metadata = first_chunk['metadata']
        
        # Create title with document info
        title = f"Document #{doc_rank}: {doc_metadata['full_path']} (Best match: {first_chunk['score']:.2%})"
        
        group_box = CollapsibleGroupBox(title)
        
        # Add summary info
        summary = QLabel(f"📄 {len(chunks)} relevant section(s) found in this document")
        summary.setFont(QFont("Arial", 10))
        summary.setStyleSheet("color: #666; margin: 5px;")
        group_box.addWidget(summary)
        
        # Add each chunk
        for i, chunk in enumerate(chunks):
            chunk_widget = self.create_chunk_widget(i + 1, chunk)
            group_box.addWidget(chunk_widget)
        
        return group_box
    
    def create_chunk_widget(self, chunk_rank, match):
        """Create a widget for displaying a single chunk within a document"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Panel | QFrame.Raised)
        frame.setStyleSheet("QFrame { margin: 5px; padding: 8px; background-color: #f8f9fa; }")
        
        layout = QVBoxLayout(frame)
        
        # Chunk header
        header = QLabel(f"Section {chunk_rank} - Relevance: {match['score']:.2%}")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header)
        
        # Chunk position info
        chunk_info = QLabel(f"Part {match['metadata']['chunk_id'] + 1} of {match['metadata']['total_chunks']} in document")
        chunk_info.setFont(QFont("Arial", 9))
        chunk_info.setStyleSheet("color: #666;")
        layout.addWidget(chunk_info)
        
        # Preview text
        preview = QTextEdit()
        preview.setPlainText(match['metadata']['chunk_text'])
        preview.setMaximumHeight(80)  # Smaller for chunks within documents
        preview.setReadOnly(True)
        preview.setFont(QFont("Arial", 9))
        preview.setStyleSheet("""
        QTextEdit { 
            background-color: #ffffff; 
            color: #000000; 
            border: 1px solid #cccccc;
            padding: 5px;
        }
    """)
        layout.addWidget(preview)
        
        return frame
    
    def handle_error(self, error_msg):
        self.search_button.setEnabled(True)
        self.status_label.setText(f"❌ Error: {error_msg}")

def main():
    app = QApplication(sys.argv)
    window = DocumentSearchApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()