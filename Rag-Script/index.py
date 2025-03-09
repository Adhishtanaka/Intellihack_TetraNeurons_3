import os
import asyncio
import chainlit as cl
from llama_cpp import Llama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


model_path = r"../unsloth.Q4_K_M.gguf"

vector_store = None


def load_documents():
    folder_path = "documents"  
    docs = []
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt") or filename.endswith(".md"): 
                file_path = os.path.join(folder_path, filename)
                loader = TextLoader(file_path)
                documents = loader.load()
                docs.extend(text_splitter.split_documents(documents))
        
        return docs
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# Initialize gguf model 
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=6,
    n_batch=512, 
    verbose=True
)

def initialize_vector_store(docs):
    try:
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(docs, embedding=embeddings, persist_directory="./chroma_db")
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Initializing RAG system...").send()
    docs = load_documents()
    if not docs:
        await cl.Message(content="Failed to load documents. Please check your file paths.").send()
        return
    global vector_store
    vector_store = initialize_vector_store(docs)
    if not vector_store:
        await cl.Message(content="Failed to initialize vector store.").send()
        return
    await cl.Message(content="RAG system initialized successfully! Ask me anything about your documents.").send()

@cl.on_message
async def on_message(message):
    global vector_store
    if not vector_store:
        await cl.Message(content="Vector store not initialized. Please restart the chat.").send()
        return

    query = message.content
    docs = vector_store.similarity_search(query, k=2)

    context = "\n\n".join([doc.page_content for doc in docs])

    rag_prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information, answer the following query concisely. If the context doesn't contain relevant information, 
    clearly state that the information is not in the provided documents.
    
    USER: {query}
    ASSISTANT:
    """

    # Run Llama model asynchronously
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: llm(
        rag_prompt,
        max_tokens=1024,
        temperature=0.5,
        stop=["USER:"]
    ))

    # Send the final cleaned response
    answer = response["choices"][0]["text"].strip()
    await cl.Message(content=answer).send()

    