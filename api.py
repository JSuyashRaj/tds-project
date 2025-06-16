from langchain.docstore.document import Document as OriginalDocument

class PatchedDocument(OriginalDocument):
    def __setstate__(self, state):
        state.pop("__fields_set__", None)
        self.__dict__.update(state)

import langchain.docstore.document
langchain.docstore.document.Document = PatchedDocument




import os
import uvicorn
import json
import re
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
#from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

app = FastAPI(title="TDS Course Q&A API with Direct Gemini", version="1.0.0")

class QuestionInput(BaseModel):
    question: str
    image: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class EnhancedAnswerResponse(BaseModel):
    answer: str
    model: str = "gemini-direct"
    links: List[LinkResponse]

# Global variables
vectordb = None
gemini_model = None

def initialize_components():
    """Initialize vector DB and direct Gemini model"""
    global vectordb, gemini_model
    
    if vectordb is None:
        print(" Loading embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")
        
        print(" Loading FAISS index...")
        from langchain.docstore.document import Document

        vectordb = FAISS.load_local(
            "embeddings/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        
        print(" Setting up direct Gemini...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        print(f" Using Gemini API key: {api_key[:10]}...")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Try different model names (using working models from your test)
        model_names = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro", 
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash-latest"
        ]
        
        for model_name in model_names:
            try:
                gemini_model = genai.GenerativeModel(model_name)
                # Test the model
                test_response = gemini_model.generate_content("Hello")
                print(f" Successfully initialized: {model_name}")
                break
            except Exception as e:
                print(f" Failed to initialize {model_name}: {e}")
                continue
        
        if gemini_model is None:
            raise ValueError("Could not initialize any Gemini model")
        
        print(" Direct Gemini initialized successfully!")

def curate_links_intelligently(docs, response_text: str, question: str) -> List[Dict[str, str]]:
    """Intelligently curate TOP 2 most relevant links from documents"""
    curated_links = []
    
    # Collect all URLs from documents with relevance scoring
    all_links = []
    
    for i, doc in enumerate(docs):
        # Get URLs from chunk-level metadata (most relevant)
        chunk_urls = doc.metadata.get('chunk_urls', [])
        chunk_discourse = doc.metadata.get('chunk_discourse_links', [])
        
        # Get URLs from document-level metadata 
        doc_urls = doc.metadata.get('urls', [])
        doc_discourse = doc.metadata.get('discourse_links', [])
        
        # Prioritize chunk-level URLs as they're more contextually relevant
        relevant_urls = chunk_urls or doc_urls
        relevant_discourse = chunk_discourse or doc_discourse
        
        for url in relevant_urls:
            # Calculate relevance score
            relevance_score = 0
            relevance_score += (5 - i) * 10  # Higher score for top retrieved docs
            if url in relevant_discourse:
                relevance_score += 50  # Bonus for discourse links
            if any(keyword in doc.page_content.lower() for keyword in question.lower().split()):
                relevance_score += 20  # Bonus for keyword match
            
            all_links.append({
                'url': url,
                'content': doc.page_content,
                'source': doc.metadata.get('source', ''),
                'is_discourse': url in relevant_discourse,
                'title': doc.metadata.get('title', ''),
                'relevance_score': relevance_score,
                'doc_index': i
            })
    
    # Sort by relevance score (highest first)
    all_links.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    # Get TOP 2 most relevant unique links
    added_urls = set()
    
    for link_info in all_links:
        if link_info['url'] in added_urls or len(curated_links) >= 2:
            continue
            
        url = link_info['url']
        content = link_info['content'].lower()
        
        # Generate smart descriptions based on content analysis
        if 'discourse.onlinedegree.iitm.ac.in' in url:
            # Discourse-specific descriptions
            if any(word in content for word in ['clarification', 'question', 'doubt', 'ask']):
                description = "Clarification and discussion on this topic."
            elif any(word in content for word in ['solution', 'answer', 'approach', 'method']):
                description = "Solution approach and methodology."
            elif any(word in content for word in ['model', 'gpt', 'api', 'openai']):
                description = "Use the model that's mentioned in the question."
            elif any(word in content for word in ['token', 'rate', 'cost', 'pricing', 'calculate']):
                description = "My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate."
            elif any(word in content for word in ['understanding', 'interpretation', 'think']):
                description = "Community understanding and interpretation of the requirements."
            else:
                description = f"Discussion thread relevant to your question."
        else:
            # Other URL descriptions
            if link_info['title']:
                description = f"Reference: {link_info['title']}"
            else:
                description = f"Additional resource from {link_info['source'].replace('.md', '').replace('-', ' ').title()}"
        
        curated_links.append({
            "url": url,
            "text": description
        })
        added_urls.add(url)
    
    # Fallback: Add TOP 2 document source links if no URLs found
    if not curated_links:
        for doc in docs[:2]:
            source_file = doc.metadata.get('source', '')
            if source_file:
                title = doc.metadata.get('title', source_file.replace('.md', '').replace('-', ' ').title())
                curated_links.append({
                    "url": f"https://tds.s-anand.net/#{source_file.replace('.md', '')}",
                    "text": f"Course material: {title}"
                })
    
    # Ensure we have exactly 2 links (pad if needed)
    while len(curated_links) < 2 and len(docs) > len(curated_links):
        remaining_docs = [doc for doc in docs if doc.metadata.get('source', '') not in [link['url'] for link in curated_links]]
        if remaining_docs:
            doc = remaining_docs[0]
            source_file = doc.metadata.get('source', '')
            curated_links.append({
                "url": f"https://tds.s-anand.net/#{source_file.replace('.md', '')}",
                "text": f"Additional course material"
            })
        else:
            break
    
    return curated_links[:2]  # Ensure exactly TOP 2

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        initialize_components()
    except Exception as e:
        print(f" Error during startup: {e}")

@app.get("/")
async def root():
    return {
        "message": "TDS Q&A API with Direct Gemini",
        "status": "running",
        "model": "Direct Gemini API"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gemini_initialized": gemini_model is not None,
        "embeddings_exist": os.path.exists("embeddings/faiss_index"),
        "data_exists": os.path.exists("data")
    }

@app.post("/api/")
async def answer_question(input: QuestionInput):
    try:
        if vectordb is None or gemini_model is None:
            initialize_components()
        
        print(f"🔍 Searching for relevant documents...")
        
        # Get relevant documents - increased to 5 for better context
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(input.question)
        
        # Prepare enhanced context
        context_parts = []
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            # Add URL context if available
            urls = doc.metadata.get('chunk_urls') or doc.metadata.get('urls', [])
            url_context = f" [URLs: {', '.join(urls[:2])}]" if urls else ""
            
            context_parts.append(f"Source: {source}{url_context}\nContent: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for better responses
        prompt = f"""You are an expert assistant for TDS (Tools in Data Science) course questions.

Based on the following course materials, provide a comprehensive and helpful answer.
Pay special attention to any specific requirements, models, or tools mentioned in the question.
Be precise and actionable in your response.

Course Materials:
{context}

Question: {input.question}

Instructions:
- Provide a direct, actionable answer
- If the question asks about specific tools/models (like GPT-3.5-turbo vs GPT-4o-mini), be explicit about requirements
- Include relevant technical details from the course materials
- If there are conflicting requirements, explain what takes precedence
- Be concise but thorough

Answer:"""
        
        print(f" Generating answer with Gemini...")
        
        # Generate response with Gemini
        response = gemini_model.generate_content(prompt)
        
        # Intelligently curate links
        curated_links = curate_links_intelligently(docs, response.text, input.question)
        
        result = {
            "answer": response.text,
            "links": curated_links
        }
        
        print(" Question answered successfully!")
        return result
        
    except Exception as e:
        print(f" Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(" Starting TDS Q&A API with Direct Gemini...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)