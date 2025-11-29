import os
# import qdrant packages
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
# import langchain packages
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI  # Added for Novita
from config import (
    QDRANT_HOST, QDRANT_PORT,
    COLLECTION_NAME, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    HF_TOKEN, LLAMA_MODEL
)
import uuid


class RAGEngine:
    def __init__(self):
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        
        # Create collection if not exists
        self._init_collection()
        
        # Load or ingest documents
        self.documents = self._load_or_ingest()
        
        # LLM - Using Hugging Face Router with OpenAI-compatible endpoint
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model=LLAMA_MODEL,
            openai_api_key=HF_TOKEN,
            openai_api_base="https://router.huggingface.co/v1",  # Hugging Face Router endpoint
            temperature=0.1,
            max_tokens=300
        )
        
        # Create prompt template
        system_prompt = (
            "You are a helpful assistant for xyz real estate company. "
            "Use the following context to answer the question. The context may include "
            "property listings in CSV format with details like ID, city, property type, price, beds/baths, and features. "
            "When asked about available properties, extract and present the information in a clear, readable format. "
            "If you don't know the answer, say so.\n\n"
            "Context: {context}"
        )
        
        
        # Create prompt template WITH MessagesPlaceholder for history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create RAG chain using LCEL
        def get_context(inputs: dict) -> str:
            """Retrieve context for the question"""
            question = inputs["question"]
            docs = self._search_similar(question)
            return "\n\n".join([doc["content"] for doc in docs])
        
        # Chain without history wrapper
        self.chain = (
            RunnablePassthrough.assign(context=get_context)
            | prompt
            | llm
            | StrOutputParser()
        )

        # Store for chat histories (manual)
        self.store = {}
    
    def _init_collection(self):
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            
            # Optionally recreate collection (set to True if you want fresh data)
            FORCE_RECREATE = True  # Change to False after first run
            
            if COLLECTION_NAME in collections and FORCE_RECREATE:
                print(f"Deleting existing collection: {COLLECTION_NAME}")
                self.client.delete_collection(COLLECTION_NAME)
                collections.remove(COLLECTION_NAME)
            
            if COLLECTION_NAME not in collections:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=384,  # Matches HuggingFace sentence-transformers embedding size
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {COLLECTION_NAME}")
            else:
                print(f"Collection {COLLECTION_NAME} already exists")
        except Exception as e:
            print(f"Error initializing collection: {e}")
    
    def _load_or_ingest(self):
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Load all documents from data directory
        documents = []
        data_path = "data"
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f"Created {data_path} directory. Please add documents.")
            return []
        
        for filename in os.listdir(data_path):
            file_path = os.path.join(data_path, filename)
            if os.path.isfile(file_path):
                if not filename.endswith('.txt'):
                    print(f"Skipping non-txt file: {filename}")
                    continue
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded: {filename} ({len(docs)} documents)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        if not documents:
            print("No documents found in data directory")
            return []
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Check if collection already has documents
        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            if collection_info.points_count > 0:
                print(f"Collection already has {collection_info.points_count} points.")
                # Still return chunks for reference
                return chunks
        except Exception as e:
            print(f"Error checking collection: {e}")
        
        # Embed and upload chunks to Qdrant
        print("Ingesting documents into Qdrant...")
        points = []
        for idx, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_model.embed_query(chunk.page_content)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
            )
            points.append(point)
            
            # Upload in batches of 100
            if len(points) >= 100:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                print(f"Uploaded {idx + 1}/{len(chunks)} chunks")
                points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            print(f"Uploaded all {len(chunks)} chunks")
        
        return chunks
    
    def _search_similar(self, query: str, k: int = None):
        """Search for similar documents using Qdrant"""
        if k is None:
            k = TOP_K
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in Qdrant - use query_points instead of search
            search_results = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=k
            ).points
            
            # Format results - handle different payload structures
            docs = []
            for result in search_results:
                # Check if payload has 'content' or 'page_content'
                content = result.payload.get("content") or result.payload.get("page_content", "")
                
                docs.append({
                    "content": content,
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score
                })
            
            return docs
        except Exception as e:
            print(f"Error searching: {e}")
            import traceback
            print(traceback.format_exc())
            return []
    
    def ask(self, question: str, session_id: str = "default") -> str:
        """Ask a question and get an answer based on the documents"""
        if not self.documents:
            return "No documents loaded. Please add .txt documents to the data directory."
        
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Get or create chat history for this session
            if session_id not in self.store:
                self.store[session_id] = []
            
            chat_history = self.store[session_id]
            
            # Invoke chain with question and chat history
            response = self.chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            # Save this exchange to history
            self.store[session_id].append(HumanMessage(content=question))
            self.store[session_id].append(AIMessage(content=response))
            
            return response
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return f"Error processing question: {str(e)}"
    
    def ask_with_sources(self, question: str, session_id: str = "default") -> dict:
        """Ask a question and return answer with source documents"""
        if not self.documents:
            return {"answer": "No documents loaded", "sources": []}
        
        try:
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Get or create chat history for this session
            if session_id not in self.store:
                self.store[session_id] = []
            
            chat_history = self.store[session_id]
            
            # Get relevant documents
            docs = self._search_similar(question)
            
            # Get answer
            answer = self.chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            # Save this exchange to history
            self.store[session_id].append(HumanMessage(content=question))
            self.store[session_id].append(AIMessage(content=answer))
            
            return {
                "answer": answer,
                "sources": [
                    {
                        "content": doc["content"][:200] + ("..." if len(doc["content"]) > 200 else ""),
                        "metadata": doc["metadata"],
                        "score": doc["score"]
                    }
                    for doc in docs
                ]
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in ask_with_sources: {error_details}")
            return {"answer": f"Error: {str(e)}", "sources": []}