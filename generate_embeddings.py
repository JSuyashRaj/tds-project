import os
import glob
import markdown
import faiss
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def extract_urls_from_markdown(content):
    """Extract URLs from markdown content"""
    url_pattern = r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+'
    urls = re.findall(url_pattern, content)
    return [url if url.startswith('http') else f'https://{url}' for url in urls]

def load_and_convert_markdown(folder):
    docs = []
    for file in glob.glob(os.path.join(folder, "*.md")):
        with open(file, encoding="utf-8") as f:
            raw_content = f.read()
            
            # Extract URLs from raw markdown before conversion
            urls = extract_urls_from_markdown(raw_content)
            
            # Convert to HTML then to text
            html = markdown.markdown(raw_content)
            text = ''.join(BeautifulSoup(html, 'html.parser').stripped_strings)
            
            # Create enhanced metadata
            metadata = {
                "source": os.path.basename(file),
                "raw_content": raw_content,  # Keep raw content for URL extraction
                "urls": urls,
                "url_count": len(urls)
            }
            
            # Add discourse links specifically
            discourse_urls = [url for url in urls if 'discourse.onlinedegree.iitm.ac.in' in url]
            if discourse_urls:
                metadata["discourse_links"] = discourse_urls
            
            # Extract title from markdown
            title_match = re.search(r'^#\s+(.+)$', raw_content, re.MULTILINE)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
            
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

def enhance_chunks_with_urls(docs):
    """Add URL information to individual chunks"""
    enhanced_docs = []
    
    for doc in docs:
        # Get URLs from the original document's raw content
        if 'raw_content' in doc.metadata:
            chunk_start = doc.page_content[:100]  # First 100 chars of chunk
            raw_content = doc.metadata['raw_content']
            
            # Find approximate position in raw content
            chunk_position = raw_content.find(chunk_start)
            
            # Extract URLs near this chunk (within 500 chars)
            if chunk_position >= 0:
                start_pos = max(0, chunk_position - 250)
                end_pos = min(len(raw_content), chunk_position + len(doc.page_content) + 250)
                chunk_context = raw_content[start_pos:end_pos]
                
                chunk_urls = extract_urls_from_markdown(chunk_context)
                if chunk_urls:
                    doc.metadata['chunk_urls'] = chunk_urls
                    
                # Check for discourse links in this chunk
                discourse_urls = [url for url in chunk_urls if 'discourse.onlinedegree.iitm.ac.in' in url]
                if discourse_urls:
                    doc.metadata['chunk_discourse_links'] = discourse_urls
        
        enhanced_docs.append(doc)
    
    return enhanced_docs

if __name__ == "__main__":
    print("Loading markdown files...")
    raw_docs = load_and_convert_markdown("data")
    print(f"Loaded {len(raw_docs)} documents")

    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(raw_docs)
    print(f"Created {len(docs)} chunks")

    print("Enhancing chunks with URL information...")
    enhanced_docs = enhance_chunks_with_urls(docs)

    print(f"Generating embeddings for {len(enhanced_docs)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(enhanced_docs, embeddings)

    os.makedirs("embeddings", exist_ok=True)
    vectordb.save_local("embeddings/faiss_index")
    print("Enhanced embedding index saved to embeddings/faiss_index")
    
    # Print some statistics
    total_urls = sum(len(doc.metadata.get('urls', [])) for doc in raw_docs)
    discourse_count = sum(1 for doc in raw_docs if 'discourse_links' in doc.metadata)
    
    print(f"\nStatistics:")
    print(f"- Total URLs found: {total_urls}")
    print(f"- Documents with discourse links: {discourse_count}")
    print(f"- Enhanced chunks created: {len(enhanced_docs)}")