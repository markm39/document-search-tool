import streamlit as st
import os
from collections import defaultdict
from drive_to_vector_pipeline import DriveToVectorPipeline

st.set_page_config(
    page_title="Document Search Tool", 
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
    font-weight: 600;
}
.search-result {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin-bottom: 1rem;
}
.document-header {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}
.relevance-score {
    color: #28a745;
    font-weight: bold;
}
.content-preview {
    background-color: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #495057;
}
.pdf-warning {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 4px;
    padding: 8px;
    margin-bottom: 8px;
    font-size: 0.85em;
    color: #856404;
}
.professional-icon {
    color: #6c757d;
    margin-right: 5px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Document Search Platform</h1>', unsafe_allow_html=True)
st.markdown("**AI-powered semantic search for your Google Drive documents**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input with better security messaging
    st.markdown("### Authentication")
    api_key = st.text_input(
        "Pinecone API Key", 
        type="password",
        help="Your API key is encrypted and not stored permanently",
        placeholder="Enter your Pinecone API key"
    )
    
    if api_key:
        st.success("✓ API key provided")
    
    # Search Settings
    st.markdown("### Search Parameters")
    
    max_results = st.selectbox(
        "Maximum Results",
        options=[5, 10, 15, 20, 30],
        index=1,  # Default to 10
        help="Total number of document sections to return"
    )
    
    max_documents = st.selectbox(
        "Maximum Documents", 
        options=[3, 5, 8, 10],
        index=1,  # Default to 5
        help="Maximum number of different documents to display"
    )
    
    show_full_content = st.checkbox(
        "Show Complete Content",
        value=True,
        help="Display full text content instead of preview excerpts"
    )
    
    # Instructions
    st.markdown("---")
    st.markdown("### Usage Instructions")
    st.markdown("""
    1. **Authenticate**: Enter your Pinecone API key above
    2. **Configure**: Adjust search parameters as needed
    3. **Initialize**: Wait for system initialization
    4. **Search**: Use natural language queries
    5. **Review**: Results are grouped by document with relevance scores
    """)
    
    st.markdown("---")
    st.markdown("### System Information")
    st.markdown("**Version:** 1.1.0")
    st.markdown("**Technology Stack:** Streamlit, Pinecone, Sentence Transformers")

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialization_error = None

# Initialize pipeline if API key is provided
if api_key and not st.session_state.pipeline and not st.session_state.initialization_error:
    with st.spinner("Initializing search system..."):
        try:
            st.session_state.pipeline = DriveToVectorPipeline(
                pinecone_api_key=api_key,
                index_name="drive-docs-rag",
                search_only=True
            )
            st.success("Search system ready")
            st.rerun()
        except Exception as e:
            st.session_state.initialization_error = str(e)
            st.error(f"Initialization failed: {e}")

# Helper function to clean and format text
def format_content(text, mime_type):
    """Clean and format content for better display"""
    if not text:
        return "No content available"
    
    # Remove excessive whitespace and normalize line breaks
    import re
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Replace multiple newlines with double newlines (paragraph breaks)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Special handling for PDFs (often have weird spacing)
    if mime_type == 'application/pdf':
        # Try to fix broken words (common in PDF extraction)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Fix sentences split across lines
        text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', text)
        # Clean up any remaining artifacts
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Main search interface
if st.session_state.pipeline:
    # Search input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query (Description of the document you're looking for)",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    # Perform search
    if (query and search_button) or (query and st.session_state.get('auto_search', False)):
        with st.spinner("Searching documents..."):
            try:
                # Get search results with user-specified limit
                results = st.session_state.pipeline.search_documents(query, top_k=max_results)
                
                if results['matches']:
                    # Group results by document
                    def group_results_by_document(matches):
                        document_groups = defaultdict(list)
                        
                        for match in matches:
                            file_id = match['metadata']['file_id']
                            document_groups[file_id].append(match)
                        
                        # Sort chunks within each document by score and take top 3
                        for file_id in document_groups:
                            document_groups[file_id].sort(key=lambda x: x['score'], reverse=True)
                            document_groups[file_id] = document_groups[file_id][:3]
                        
                        # Sort documents by their best match score and limit to user preference
                        sorted_docs = sorted(
                            document_groups.items(),
                            key=lambda x: x[1][0]['score'],
                            reverse=True
                        )[:max_documents]
                        
                        return dict(sorted_docs)
                    
                    grouped_results = group_results_by_document(results['matches'])
                    
                    # Display results summary
                    total_docs = len(grouped_results)
                    total_chunks = sum(len(chunks) for chunks in grouped_results.values())
                    
                    st.markdown(f"### Search Results")
                    st.info(f"Found **{total_chunks}** relevant sections across **{total_docs}** documents")
                    
                    # Display grouped results
                    for doc_rank, (file_id, chunks) in enumerate(grouped_results.items(), 1):
                        first_chunk = chunks[0]
                        doc_metadata = first_chunk['metadata']
                        
                        # Document header
                        with st.expander(
                            f"Document #{doc_rank}: {doc_metadata['file_name']} "
                            f"(Relevance: {first_chunk['score']:.1%})",
                            expanded=doc_rank <= 2  # Expand first 2 results
                        ):
                            # Document info
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**File Path:** `{doc_metadata['full_path']}`")
                                st.markdown(f"**File Type:** {doc_metadata.get('mime_type', 'Unknown')}")
                            with col2:
                                st.markdown(f"**Relevant Sections:** {len(chunks)}")
                                if doc_metadata.get('modified_time'):
                                    st.markdown(f"**Last Modified:** {doc_metadata['modified_time'][:10]}")
                            
                            # Display each chunk
                            for i, chunk in enumerate(chunks):
                                st.markdown(f"**Section {i+1}** - Relevance Score: {chunk['score']:.1%}")
                                st.markdown(f"*Part {chunk['metadata']['chunk_id'] + 1} of {chunk['metadata']['total_chunks']} in document*")
                                
                                # Show PDF warning if applicable
                                if chunk['metadata'].get('mime_type') == 'application/pdf':
                                    st.markdown("""
                                    <div class="pdf-warning">
                                    <strong>PDF Content Notice:</strong> Text extracted from PDF documents may contain formatting irregularities due to the extraction process.
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Content display
                                if show_full_content:
                                    # Get the full chunk content and format it
                                    full_content = chunk['metadata'].get('chunk_text', '')
                                    
                                    # If chunk_text is truncated (ends with "..."), show that it's partial
                                    if full_content.endswith('...'):
                                        st.markdown("*Note: This content may be truncated due to storage limitations.*")
                                    
                                    formatted_content = format_content(full_content, chunk['metadata'].get('mime_type'))
                                    formatted_html = formatted_content.replace('\n', '<br>')
                                    
                                    st.markdown(f"""
                                    <div class="content-preview">
                                    {formatted_html}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    # Show preview (original behavior)
                                    preview_text = chunk['metadata']['chunk_text'][:400] + "..." if len(chunk['metadata']['chunk_text']) > 400 else chunk['metadata']['chunk_text']
                                    formatted_preview = format_content(preview_text, chunk['metadata'].get('mime_type'))
                                    formatted_html_preview = formatted_preview.replace('\n', '<br>')
                                    
                                    st.markdown(f"""
                                    <div class="content-preview">
                                    {formatted_html_preview}
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Show content stats
                                st.caption(f"Content length: {chunk['metadata'].get('chunk_length', 'Unknown')} characters")
                                
                                if i < len(chunks) - 1:
                                    st.markdown("---")
                
                else:
                    st.warning("No results found. Please try different search terms or verify that your documents are indexed.")
                    
                    # Suggestions for better search
                    with st.expander("Search Optimization Tips"):
                        st.markdown("""
                        **Improve your search results:**
                        - Include contextual information: "meeting notes Chicago"
                        - Use complete phrases
                        - Be descriptive: "Q4 sales performance report"
                        """)
                    
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.exception(e)  # Show full error for debugging

elif api_key and st.session_state.initialization_error:
    st.error(f"Cannot perform search: {st.session_state.initialization_error}")
    if st.button("Retry Initialization"):
        st.session_state.initialization_error = None
        st.rerun()

elif not api_key:
    st.info("Please provide your Pinecone API key in the sidebar to begin")
    
    # Help section
    with st.expander("Getting Started Guide"):
        st.markdown("""
        ### Obtaining Your Pinecone API Key
        1. Visit [Pinecone.io](https://www.pinecone.io/) and create an account
        2. Create a new project in your dashboard
        3. Navigate to project settings and copy your API key
        4. Enter the key in the sidebar authentication section
        
        ### Initial Setup Requirements
        - The system connects to pre-indexed Google Drive documents
        - If indexing is required, contact your system administrator
        - Documentation available at: [GitHub Repository](https://github.com/markm39/document-search-tool)
        
        ### Search Best Practices
        - Use natural language queries for optimal results
        - Be specific about the information you're seeking
        - Results are ranked by semantic similarity, not keyword matching
        - Adjust search parameters in the sidebar as needed
        """)