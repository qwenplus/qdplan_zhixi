"""
RAG问答引擎 (LLM应用层)
========================
基于检索增强生成(RAG)的智能问答系统。
支持OpenAI API和本地Ollama两种模式。

技术栈:
    - LangChain: LLM应用编排框架
    - ChromaDB: 向量数据库 (嵌入式)
    - OpenAI / Ollama: 大语言模型

使用示例:
    engine = RAGEngine()
    engine.ingest_documents(chunks)   # 导入文档块
    answer = engine.ask("这篇论文的主要贡献是什么？")
    print(answer["answer"])
    print(answer["sources"])
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGAnswer:
    """RAG问答结果"""
    question: str
    answer: str
    sources: list = field(default_factory=list)  # 来源文档块
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "model_used": self.model_used,
        }


class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) 问答引擎

    工作流程:
    1. 文档切块 → Embedding → 存入ChromaDB
    2. 用户提问 → 检索相关文档块 → 构建Prompt → LLM生成回答

    支持两种运行模式:
    - OpenAI模式: 使用GPT系列模型 (需要API Key)
    - Ollama模式: 使用本地开源模型 (零成本)

    Args:
        use_ollama: 是否使用Ollama本地模型
        ollama_base_url: Ollama服务地址
        ollama_model: Ollama模型名称
        openai_model: OpenAI模型名称
        embedding_model: Embedding模型名称
        collection_name: ChromaDB集合名称
        persist_directory: ChromaDB持久化目录
    """

    def __init__(
        self,
        use_ollama: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:7b",
        ollama_embedding_model: Optional[str] = None,
        openai_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        hf_embedding_model: Optional[str] = None,
        collection_name: str = "zhixi_docs",
        persist_directory: str = "data/processed/chroma_db",
    ):
        self.use_ollama = use_ollama
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.openai_model = openai_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")

        # Embedding 配置: 支持三种模式 ollama / huggingface / openai
        self.embedding_provider = embedding_provider or os.getenv("EMBEDDING_PROVIDER", "ollama" if use_ollama else "openai")
        self.ollama_embedding_model = ollama_embedding_model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.hf_embedding_model = hf_embedding_model or os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 延迟初始化
        self._llm = None
        self._embeddings = None
        self._vectorstore = None
        self._qa_chain = None

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

    def _init_llm(self):
        """初始化LLM模型"""
        if self._llm is not None:
            return

        if self.use_ollama:
            from langchain_community.llms import Ollama
            self._llm = Ollama(
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                temperature=0.1,
            )
            print(f"[RAG] 使用Ollama模型: {self.ollama_model}")
        else:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.openai_model,
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print(f"[RAG] 使用OpenAI模型: {self.openai_model}")

    def _init_embeddings(self):
        """
        初始化Embedding模型

        支持三种模式 (通过 EMBEDDING_PROVIDER 环境变量控制):
        - ollama:     使用Ollama本地embedding模型 (如 nomic-embed-text)
        - huggingface: 使用HuggingFace sentence-transformers (如 bge-small-zh-v1.5，中文推荐)
        - openai:     使用OpenAI Embedding API (如 text-embedding-3-small)
        """
        if self._embeddings is not None:
            return

        provider = self.embedding_provider.lower()

        if provider == "ollama":
            from langchain_community.embeddings import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model=self.ollama_embedding_model,
            )
            print(f"[RAG] Embedding使用Ollama: {self.ollama_embedding_model}")

        elif provider == "huggingface":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.hf_embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
            print(f"[RAG] Embedding使用HuggingFace: {self.hf_embedding_model}")

        else:  # openai
            from langchain_openai import OpenAIEmbeddings
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print(f"[RAG] Embedding使用OpenAI: {self.embedding_model}")

    def _init_vectorstore(self):
        """初始化向量数据库"""
        if self._vectorstore is not None:
            return

        self._init_embeddings()

        import chromadb
        from langchain_community.vectorstores import Chroma

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"[RAG] ChromaDB已初始化: {self.collection_name}")

    def ingest_documents(self, chunks: list):
        """
        将文档块导入向量数据库

        Args:
            chunks: [{"text": "...", "page": 1, "chunk_id": 0}, ...]
                    来自 DocumentParser.get_text_chunks() 的输出
        """
        if not chunks:
            print("[RAG] 没有文档块可导入")
            return

        self._init_vectorstore()

        # 准备LangChain格式的文档
        from langchain_core.documents import Document

        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata={
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                    "source": chunk.get("source", "unknown"),
                },
            )
            documents.append(doc)

        # 批量添加到ChromaDB
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._vectorstore.add_documents(batch)
            print(f"[RAG] 已导入 {min(i + batch_size, len(documents))}/{len(documents)} 个文档块")

        print(f"[RAG] 文档导入完成，共 {len(documents)} 个块")

    def ask(
        self,
        question: str,
        top_k: int = 4,
        include_sources: bool = True,
    ) -> RAGAnswer:
        """
        基于文档知识库回答问题

        Args:
            question: 用户问题
            top_k: 检索的文档块数量
            include_sources: 是否返回来源信息

        Returns:
            RAGAnswer: 包含答案和来源的结果
        """
        self._init_llm()
        self._init_vectorstore()

        # 1. 检索相关文档块
        relevant_docs = self._vectorstore.similarity_search(
            question, k=top_k
        )

        if not relevant_docs:
            return RAGAnswer(
                question=question,
                answer="抱歉，未在文档中找到相关信息。请确保已导入文档。",
                sources=[],
                model_used=self.openai_model if not self.use_ollama else self.ollama_model,
            )

        # 2. 构建上下文
        context_parts = []
        sources = []
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"[文档片段 {i + 1}]\n{doc.page_content}")
            if include_sources:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "page": doc.metadata.get("page", "N/A"),
                    "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                })

        context = "\n\n".join(context_parts)

        # 3. 构建Prompt
        prompt = self._build_prompt(question, context)

        # 4. 调用LLM生成回答
        try:
            if self.use_ollama:
                answer_text = self._llm.invoke(prompt)
            else:
                from langchain_core.messages import HumanMessage, SystemMessage
                response = self._llm.invoke([
                    SystemMessage(content="你是一个专业的文档分析助手，基于提供的文档内容回答问题。"),
                    HumanMessage(content=prompt),
                ])
                answer_text = response.content
        except Exception as e:
            answer_text = f"LLM调用出错: {e}"

        model_name = self.ollama_model if self.use_ollama else self.openai_model

        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            model_used=model_name,
        )

    def _build_prompt(self, question: str, context: str) -> str:
        """构建RAG Prompt"""
        return f"""请基于以下文档内容回答用户的问题。

要求:
1. 仅基于文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、简洁、有条理
4. 如果涉及数据，请引用具体数字

文档内容:
{context}

用户问题: {question}

回答:"""

    def search(self, query: str, top_k: int = 5) -> list:
        """
        仅检索相关文档块 (不生成回答)

        Args:
            query: 搜索查询
            top_k: 返回数量

        Returns:
            list of dict: 检索结果
        """
        self._init_vectorstore()

        docs = self._vectorstore.similarity_search(query, k=top_k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            })
        return results

    def clear_collection(self):
        """清空向量数据库集合"""
        self._init_vectorstore()
        self._vectorstore.delete_collection()
        # 重新初始化
        self._vectorstore = None
        print("[RAG] 向量数据库已清空")


# ===== 便捷函数 =====

def quick_rag(
    pdf_path: str,
    question: str,
    use_ollama: bool = False,
) -> RAGAnswer:
    """
    一键RAG: 解析PDF → 导入 → 问答

    Args:
        pdf_path: PDF文件路径
        question: 问题
        use_ollama: 是否使用本地Ollama模型

    Returns:
        RAGAnswer: 问答结果
    """
    from src.doc_parser import DocumentParser

    # 解析文档
    parser = DocumentParser(pdf_path)
    chunks = parser.get_text_chunks(chunk_size=500, chunk_overlap=50)

    # 初始化RAG引擎
    engine = RAGEngine(use_ollama=use_ollama)
    engine.ingest_documents(chunks)

    # 问答
    return engine.ask(question)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        pdf_file = sys.argv[1]
        question = " ".join(sys.argv[2:])
        print(f"文档: {pdf_file}")
        print(f"问题: {question}\n")

        result = quick_rag(pdf_file, question)
        print(f"回答:\n{result.answer}")
        print(f"\n模型: {result.model_used}")
        print(f"来源数: {len(result.sources)}")
    else:
        print("用法: python rag_engine.py <pdf文件路径> <问题>")
        print("示例: python rag_engine.py paper.pdf 这篇论文的主要贡献是什么？")
