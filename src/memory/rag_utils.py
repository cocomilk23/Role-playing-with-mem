import os
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from .persistence import ProfessionalMemoryRAG
from .types import ProfessionalMemory, ProfessionalMemoryResult

# 默认使用 SentenceTransformer 的 all-MiniLM-L6-v2 作为嵌入模型
# 注意：在沙箱环境中，由于网络限制，可能需要使用本地模型或预先下载的模型。
# 为了演示，我们使用 ChromaDB 的默认嵌入函数，它通常是 all-MiniLM-L6-v2 的轻量级版本。
# 实际生产环境应配置更强大的嵌入模型。
DEFAULT_EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class ChromaDBRAG(ProfessionalMemoryRAG):
    """
    基于 ChromaDB 的专业记忆 RAG 实现。
    """
    def __init__(self, db_path: str = "data/chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=self.db_path)

    def retrieve(self, query: str, knowledge_path: str, top_k: int = 3) -> ProfessionalMemory:
        """
        根据查询和知识路径（Collection Name）进行 RAG 检索。
        """
        try:
            collection = self.client.get_collection(
                name=knowledge_path,
                embedding_function=DEFAULT_EMBEDDING_FUNCTION
            )
        except Exception as e:
            print(f"Error getting collection {knowledge_path}: {e}")
            return ProfessionalMemory()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        professional_memory_results: List[ProfessionalMemoryResult] = []
        if results and results.get('documents'):
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                professional_memory_results.append(
                    ProfessionalMemoryResult(
                        content=doc,
                        source=meta.get('source', 'N/A'),
                        score=1.0 - dist # 简单地将距离转换为相似度得分
                    )
                )
        
        return ProfessionalMemory(results=professional_memory_results)

def index_documents_to_chroma(file_path: str, collection_name: str, db_path: str = "data/chroma_db"):
    """
    加载、分块文档，并将其索引到 ChromaDB。
    
    :param file_path: 要索引的文档路径。
    :param collection_name: ChromaDB Collection 的名称，作为角色的 knowledge_path。
    :param db_path: ChromaDB 存储路径。
    """
    print(f"--- 正在索引文档: {file_path} 到 Collection: {collection_name} ---")
    
    # 1. 加载文档 (目前仅支持 TextLoader，可扩展支持 PDF, DOCX 等)
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        return

    # 2. 分块
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # 3. 准备数据
    texts = [doc.page_content for doc in docs]
    metadatas = [{
        "source": os.path.basename(file_path),
        "chunk_index": i,
        **doc.metadata
    } for i, doc in enumerate(docs)]
    ids = [f"{collection_name}_{os.path.basename(file_path)}_{i}" for i in range(len(docs))]

    # 4. 索引到 ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    
    # 确保 Collection 存在，如果存在则清空或更新
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=DEFAULT_EMBEDDING_FUNCTION
        )
        # 简单处理：如果 Collection 存在，先删除所有内容
        collection.delete(ids=[id for id in collection.get()['ids']])
        print(f"Existing collection '{collection_name}' cleared.")
    except:
        # Collection 不存在，创建新的
        collection = client.create_collection(
            name=collection_name,
            embedding_function=DEFAULT_EMBEDDING_FUNCTION
        )
        print(f"New collection '{collection_name}' created.")

    # 5. 添加文档
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"成功索引 {len(texts)} 个文本块到 ChromaDB Collection: {collection_name}")
