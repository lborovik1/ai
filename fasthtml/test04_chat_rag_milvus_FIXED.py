"""
# test04_chat_rag_milvus_FIXED.py
# Example of chatbot using fasthtml and ollama
# Using Milvus as VectorDB
# Additional modules:
#  - RAG (Retrieval Augmented Generation)
#
# FIXES APPLIED:
# 1. Added missing numpy import
# 2. Fixed file loading to only happen once (not on every query)
# 3. Fixed collection management with proper checking
# 4. Removed hard-coded filters
# 5. Removed duplicate functions
# 6. Implemented proper unique ID management
# 7. Connected file upload to indexing
# 8. Improved text chunking with sentence-based splitting
# 9. Added comprehensive error handling
# 10. Fixed search results formatting
"""

from fasthtml.common import *
from starlette.requests import Request

import ollama
import asyncio
import numpy as np  # FIX #1: Added missing numpy import
import os
import nltk
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Download NLTK data for sentence tokenization
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

app, rt, = fast_app(live=True, ws_hdr=True)

# Simple object to store variables (replaces MyBunch from levutils)
class SimpleBag:
    pass

bag = SimpleBag()

client = ollama.Client()
model = "mistral:7b-instruct-v0.3-q4_0"
messages = []
messages_for_show = []

# Global variables
m_client = None
embedding_model = None
COLLECTION_NAME = "demo_collection"
EMBEDDING_DIM = 384

bag.script_dir = os.path.dirname(os.path.realpath(__file__))
bag.dir_out = bag.script_dir + "/uploaded_files"

sp = {"role": "system", "content": "You are a helpful and concise assistant."}

#---------------------------------------------------------------
# FIX #2 & #3: Proper initialization - create collection once
#---------------------------------------------------------------
async def init():
    """Initialize Milvus client and embedding model"""
    global m_client, embedding_model
    
    try:
        m_client = MilvusClient("./milvus_demo.db")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Check if collection exists, create if not
        if not m_client.has_collection(COLLECTION_NAME):
            m_client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=EMBEDDING_DIM
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            print(f"Collection {COLLECTION_NAME} already exists")
            
        # Create upload directory
        os.makedirs(bag.dir_out, exist_ok=True)
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    await init()

#---------------------------------------------------------------
# FIX #5: Single, clean function for reading and chunking files
# FIX #8: Improved chunking with sentence-based splitting
#---------------------------------------------------------------
def chunk_text(text, chunk_size=3):
    """
    Chunk text using sentence tokenization for better semantic boundaries
    
    Args:
        text: Input text to chunk
        chunk_size: Number of sentences per chunk
    
    Returns:
        List of text chunks
    """
    try:
        sentences = nltk.sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i+chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
        
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        # Fallback to simple splitting
        return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

def process_file_for_indexing(filename, file_path):
    """
    Process a single file: read, chunk, and create embeddings
    
    Args:
        filename: Name of the file
        file_path: Path to the file
    
    Returns:
        List of document dictionaries ready for indexing
    """
    global embedding_model
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        print(f"Processing file: {filename}, length: {len(text)} chars")
        
        # Chunk text
        chunks = chunk_text(text)
        
        if not chunks:
            print(f"Warning: No chunks created for {filename}")
            return []
        
        # Generate embeddings
        embeddings = embedding_model.encode(chunks)
        
        # FIX #6: Create unique IDs for each chunk
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = {
                "id": f"{filename}_{i}",  # Unique ID per chunk
                "vector": embedding.tolist(),
                "text": chunk,
                "filename": filename,
                "chunk_index": i
            }
            documents.append(doc)
        
        print(f"Created {len(documents)} document chunks for {filename}")
        return documents
        
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return []

#---------------------------------------------------------------
# FIX #7: Integrate file indexing with upload
#---------------------------------------------------------------
async def index_file(filename, file_path):
    """
    Index a single file into Milvus
    
    Args:
        filename: Name of the file
        file_path: Path to the file
    """
    global m_client
    
    try:
        # Process file and create document chunks
        documents = process_file_for_indexing(filename, file_path)
        
        if not documents:
            print(f"No documents to index for {filename}")
            return False
        
        # Insert into Milvus
        result = m_client.insert(
            collection_name=COLLECTION_NAME,
            data=documents
        )
        
        print(f"Successfully indexed {len(documents)} chunks from {filename}")
        return True
        
    except Exception as e:
        print(f"Error indexing file {filename}: {e}")
        return False

#---------------------------------------------------------------
# FIX #4 & #10: Proper RAG implementation without hard-coded filters
#---------------------------------------------------------------
async def do_rag(query):
    """
    Perform RAG: search for relevant documents and return context
    
    Args:
        query: User's query string
    
    Returns:
        Formatted context string from relevant documents
    """
    global m_client, embedding_model
    
    try:
        # FIX #2: Don't reload files here!
        # Generate query embedding
        query_vector = embedding_model.encode([query])[0].tolist()
        
        # Search in Milvus (FIX #4: no hard-coded filter)
        search_results = m_client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            limit=5,  # Top 5 results
            output_fields=["text", "filename", "chunk_index"]
        )
        
        # FIX #10: Extract and format results properly
        if not search_results or not search_results[0]:
            return "No relevant documents found."
        
        context_chunks = []
        for hit in search_results[0]:
            # Access entity data correctly
            text = hit['entity']['text']
            filename = hit['entity']['filename']
            context_chunks.append(f"[From {filename}]: {text}")
        
        context = "\n\n".join(context_chunks)
        return context
        
    except Exception as e:
        print(f"Error in RAG: {e}")
        return f"Error retrieving context: {str(e)}"

#---------------------------------------------------------------
@rt('/')
def get():
    """ Main page """
    bag.list_items = []
    main_page = (
        Title("Chatbot with RAG"),
        Titled('RAG Chatbot',
        Div(
            H1("Chatbot using FastHTML, Ollama & Milvus"),
            Div(
            P(Img(src="https://fastht.ml/assets/logo.svg", width=100, height=100)),
            ),
            A("About", href="/about"),
            get_history(),
            Div("File Upload:"),
                Div(P("Ask a question:"),
                Form(Group(
                     Input(id="new-prompt", type="text", name="data", placeholder="Type your question..."),
                     Button("Submit")
                     ),
                     ws_send=True, hx_ext="ws", ws_connect="/wscon", 
                     target_id='message-list',
                     hx_swap="beforeend",
                     enctype="multipart/form-data"
                     )),
                    Div("Drag files here or click to upload:", id="container", 
                    style="width: 400px; height: 150px; background-color: #e8f4f8; border: 2px dashed #4a90e2; border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer;"),
                    Form(
                    Input(id='file', name='file', type='file', multiple=True, accept=".txt", 
                          onchange="this.form.querySelector('button').click()"),
                    Button('Upload', type="submit", style="display: none;"),
                    id="upload-form",
                    hx_post="/upload",
                    target_id="container",
                    hx_swap="innerHTML",
                    enctype="multipart/form-data"
                ),
            Script(
            """
            const container = document.getElementById('container');
            const fileInput = document.getElementById('file');
            const form = document.getElementById('upload-form');
            
            container.addEventListener('click', () => fileInput.click());
            
            container.addEventListener('dragover', (event) => {
                event.preventDefault();
                container.style.backgroundColor = '#d0e8f2';
            });
            
            container.addEventListener('dragleave', () => {
                container.style.backgroundColor = '#e8f4f8';
            });

            container.addEventListener('drop', (event) => {
                event.preventDefault();
                container.style.backgroundColor = '#e8f4f8';
                const files = event.dataTransfer.files; 
                fileInput.files = files; 
                form.dispatchEvent(new Event('submit')); 
            });
            """
        )
        ))
    )
    
    return main_page

#---------------------------------------------------------------
# FIX #7: Upload now triggers indexing
# FIX #9: Added error handling
#---------------------------------------------------------------
@rt('/upload')
async def post(request: Request):
    """Handle file upload and index files immediately"""
    try:
        form = await request.form()
        uploaded_files = form.getlist("file")
        
        if not uploaded_files:
            return Div("No files uploaded", style="color: orange;")
        
        results = []
        
        for uploaded_file in uploaded_files:
            try:
                filename = uploaded_file.filename
                file_path = f"{bag.dir_out}/{filename}"
                
                # Save file
                with open(file_path, "wb") as f:
                    content = await uploaded_file.read()
                    f.write(content)
                
                # Index file immediately
                success = await index_file(filename, file_path)
                
                if success:
                    results.append(P(f"✓ {filename} uploaded and indexed", style="color: green;"))
                else:
                    results.append(P(f"⚠ {filename} uploaded but indexing failed", style="color: orange;"))
                    
            except Exception as e:
                results.append(P(f"✗ Error with {uploaded_file.filename}: {str(e)}", style="color: red;"))
        
        return Div(*results)
        
    except Exception as e:
        return Div(f"Upload error: {str(e)}", style="color: red;")

#---------------------------------------------------------------
def get_history():
    """ Get all history messages """
    listed_messages = print_all_messages()
    history = Div(listed_messages, id="chatlist")
    return history

#---------------------------------------------------------------
def add_message(data):
    """ Add message """
    i = len(messages)
    tid = f'message-{i}'
    
    list_item = Li(data, id=tid)
    bag.list_items.append(list_item)
    return list_item

#---------------------------------------------------------------
def print_all_messages():
    """ Create ul from messages and return them to main page """
    i = 0
    for message in messages_for_show:
        tid = f'message-{i}'
        list_item = Li(message['content'], id=tid)
        bag.list_items.append(list_item)
        i += 1
    
    return Ul(*bag.list_items, id='message-list')

#---------------------------------------------------------------
@rt('/about')
def get():
    """ About page """
    main_page = (
        Titled('About',
        Div(
            H1("How This RAG Chatbot Works:"),
            Div(
            P(Img(src="https://fastht.ml/assets/logo.svg", width=100, height=100)),
            ),
            P("This chatbot uses Retrieval Augmented Generation (RAG):"),
            Ul(
                Li("Upload text files to build a knowledge base"),
                Li("Files are chunked and embedded using SentenceTransformers"),
                Li("Embeddings are stored in Milvus vector database"),
                Li("When you ask a question, relevant chunks are retrieved"),
                Li("The LLM (Ollama) uses these chunks to answer your question")
            ),
            A("Home", href="/"),
        ))
    )
    return main_page

#---------------------------------------------------------------
def ChatInput():
    """ Clear the input """
    return Input(id="new-prompt", type="text", name='data',
                 placeholder="Type your question...",
                 cls="input input-bordered w-full", hx_swap_oob='true')
 
#---------------------------------------------------------------
@app.ws('/wscon')
async def ws(data:str, send):
    """ 
    WebSocket handler: processes user query with RAG and streams LLM response
    FIX #9: Added error handling
    """
    try:
        # Get relevant context from RAG
        context = await do_rag(data)
        
        messages_for_show.append({"role": "user", "content": f"{data}"})
        
        # Add context to the message for the LLM
        messages.append({"role": "user", "content": f"Context from documents:\n{context}\n\nQuestion: {data}\n\nPlease answer based on the context provided."})

        await send(
            Div(add_message(data), hx_swap_oob="beforeend", id="message-list")
        )

        # Send the clear input field command
        await send(ChatInput())

        # Model response (streaming)
        stream = ollama.chat(
            model=model,
            messages=[sp] + messages,
            stream=True,
        )
        
        # Send an empty message for the assistant response
        messages.append({"role": "assistant", "content": ""})
        
        await send(
            Div(add_message(""), hx_swap_oob="beforeend", id="message-list")
        )

        i = len(messages)
        tid = f'message-{i}'
        msg = ""

        # Stream the response
        for chunk in stream:
            chunk = chunk["message"]["content"]
            msg = msg + chunk
            messages[-1]["content"] += chunk
            await send(
                Li(chunk, id=tid, hx_swap_oob="beforeend")
            )
            await asyncio.sleep(0.01)

        messages_for_show.append({"role": "assistant", "content": f"{msg}"})
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        await send(
            Div(add_message(error_msg), hx_swap_oob="beforeend", id="message-list")
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test04_chat_rag_milvus_FIXED:app", host='localhost', port=5001, reload=True)
