"""
智析 (ZhiXi) - Streamlit Web应用
=================================
多模态文档智能分析与知识问答平台的Web界面。

启动方式:
    cd zhixi
    streamlit run src/app.py

功能:
    - 上传PDF文档并自动解析
    - NLP分析 (实体/关键词/摘要/词云)
    - 知识图谱可视化
    - RAG智能问答
"""

import os
import sys
import tempfile

# 将项目根目录加入path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ===== 页面配置 =====
st.set_page_config(
    page_title="智析 - 文档智能分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 自定义样式 =====
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    defaults = {
        "pdf_path": None,
        "doc_result": None,
        "nlp_result": None,
        "kg_built": False,
        "rag_ready": False,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.title("⚙️ 配置")

        # LLM模式选择
        llm_mode = st.radio(
            "LLM 运行模式",
            ["本地 Ollama (免费)", "OpenAI API"],
            index=0,
            help="Ollama在本地运行开源模型，零成本；OpenAI需要API Key",
        )

        if llm_mode == "OpenAI API":
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
            )
            model = st.selectbox(
                "聊天模型",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
            )
            st.session_state["use_ollama"] = False
            st.session_state["openai_model"] = model
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
        else:
            ollama_url = st.text_input(
                "Ollama 地址",
                value="http://localhost:11434",
            )
            ollama_model = st.text_input(
                "聊天模型",
                value="qwen2.5:7b",
                help="需要先运行: ollama pull qwen2.5:7b",
            )
            st.session_state["use_ollama"] = True
            st.session_state["ollama_url"] = ollama_url
            st.session_state["ollama_model"] = ollama_model

        st.divider()

        # Embedding 模型选择
        st.subheader("Embedding 模型")
        embed_provider = st.radio(
            "向量化方案",
            ["Ollama (本地，推荐)", "HuggingFace (中文推荐)", "OpenAI (付费)"],
            index=0,
            help="将文本转换为向量用于语义搜索",
        )
        if embed_provider == "Ollama (本地，推荐)":
            embed_model = st.text_input(
                "Ollama Embedding 模型",
                value="nomic-embed-text",
                help="需要先运行: ollama pull nomic-embed-text",
            )
            st.session_state["embedding_provider"] = "ollama"
            st.session_state["ollama_embedding_model"] = embed_model
        elif embed_provider == "HuggingFace (中文推荐)":
            hf_model = st.selectbox(
                "HuggingFace 模型",
                ["BAAI/bge-small-zh-v1.5", "BAAI/bge-base-zh-v1.5", "all-MiniLM-L6-v2"],
                index=0,
                help="bge-small-zh 专为中文优化，首次运行自动下载(~130MB)",
            )
            st.session_state["embedding_provider"] = "huggingface"
            st.session_state["hf_embedding_model"] = hf_model
        else:
            st.session_state["embedding_provider"] = "openai"

        st.divider()

        # RAG参数
        st.subheader("RAG 参数")
        chunk_size = st.slider("文本块大小", 200, 1000, 500, 50)
        chunk_overlap = st.slider("重叠大小", 0, 200, 50, 10)
        top_k = st.slider("检索数量", 1, 10, 4)
        st.session_state["chunk_size"] = chunk_size
        st.session_state["chunk_overlap"] = chunk_overlap
        st.session_state["top_k"] = top_k

        st.divider()
        st.caption("智析 v0.1.0 | Made with Streamlit")


def render_header():
    """渲染页面头部"""
    st.markdown('<div class="main-header">📊 智析</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">多模态文档智能分析与知识问答平台 — '
        '上传PDF，自动解析内容，构建知识图谱，智能问答</div>',
        unsafe_allow_html=True,
    )


def render_upload_section():
    """渲染文档上传区域"""
    st.subheader("📄 上传文档")

    uploaded_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        help="支持PDF格式的研究报告、论文、文档等",
    )

    if uploaded_file is not None:
        # 保存到临时文件
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["pdf_path"] = temp_path

        st.success(f"✅ 已上传: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

        # 解析按钮
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔍 解析文档", type="primary"):
                with st.spinner("正在解析文档..."):
                    _parse_document(temp_path)

    # 显示已有解析结果
    if st.session_state["doc_result"] is not None:
        _show_doc_overview()


def _parse_document(pdf_path: str):
    """解析文档"""
    try:
        from src.doc_parser import DocumentParser

        parser = DocumentParser(pdf_path)
        result = parser.parse()
        st.session_state["doc_result"] = result

        # 自动切分文本块
        chunks = parser.get_text_chunks(
            chunk_size=st.session_state.get("chunk_size", 500),
            chunk_overlap=st.session_state.get("chunk_overlap", 50),
        )
        st.session_state["chunks"] = chunks

        st.rerun()
    except Exception as e:
        st.error(f"解析出错: {e}")


def _show_doc_overview():
    """显示文档概览"""
    result = st.session_state["doc_result"]
    if result is None:
        return

    st.divider()

    # 指标卡片
    cols = st.columns(4)
    with cols[0]:
        st.metric("📄 页数", result.total_pages)
    with cols[1]:
        st.metric("📝 字符数", f"{len(result.full_text):,}")
    with cols[2]:
        table_count = sum(len(p.tables) for p in result.pages)
        st.metric("📊 表格数", table_count)
    with cols[3]:
        image_count = sum(len(p.images) for p in result.pages)
        st.metric("🖼️ 图像数", image_count)

    # 文本预览
    with st.expander("📖 查看提取的文本", expanded=False):
        st.text_area("文档全文", result.full_text[:5000], height=300)


def render_nlp_section():
    """渲染NLP分析区域"""
    if st.session_state["doc_result"] is None:
        st.info("👆 请先上传并解析文档")
        return

    st.subheader("🧠 NLP 智能分析")

    if st.button("🔬 执行分析", type="primary"):
        with st.spinner("正在执行NLP分析 (首次运行需下载模型)..."):
            _run_nlp_analysis()

    # 显示NLP结果
    if st.session_state["nlp_result"] is not None:
        _show_nlp_results()


def _run_nlp_analysis():
    """执行NLP分析"""
    try:
        from src.nlp_pipeline import NLPPipeline

        text = st.session_state["doc_result"].full_text
        nlp = NLPPipeline()
        result = nlp.analyze(
            text,
            extract_entities=True,
            extract_keywords=True,
            generate_summary=True,
        )
        st.session_state["nlp_result"] = result

        # 生成词云
        nlp.generate_wordcloud(text)

        st.rerun()
    except Exception as e:
        st.error(f"NLP分析出错: {e}")
        st.caption("提示: 首次运行需要下载模型，可能需要几分钟。确保已安装transformers和keybert。")


def _show_nlp_results():
    """显示NLP分析结果"""
    result = st.session_state["nlp_result"]
    if result is None:
        return

    tabs = st.tabs(["📌 关键词", "🏷️ 实体识别", "📝 摘要", "☁️ 词云"])

    with tabs[0]:
        if result.keywords:
            for kw, score in result.keywords:
                st.markdown(f"- **{kw}** (相关度: {score:.3f})")
        else:
            st.write("未提取到关键词")

    with tabs[1]:
        if result.entities:
            # 按类型分组显示
            from collections import defaultdict
            by_type = defaultdict(list)
            for e in result.entities:
                by_type[e.label].append(e.text)

            for entity_type, names in by_type.items():
                st.markdown(f"**{entity_type}**: {', '.join(set(names))}")
        else:
            st.write("未识别到实体")

    with tabs[2]:
        if result.summary:
            st.markdown(result.summary)
        else:
            st.write("未生成摘要")

    with tabs[3]:
        wordcloud_path = "data/processed/wordcloud.png"
        if os.path.exists(wordcloud_path):
            st.image(wordcloud_path, caption="关键词词云")
        else:
            st.write("词云未生成")


def render_kg_section():
    """渲染知识图谱区域"""
    if st.session_state["doc_result"] is None:
        st.info("👆 请先上传并解析文档")
        return

    st.subheader("🕸️ 知识图谱")

    if st.button("🔗 构建知识图谱"):
        with st.spinner("正在构建知识图谱..."):
            _build_knowledge_graph()

    if st.session_state["kg_built"]:
        _show_knowledge_graph()


def _build_knowledge_graph():
    """构建知识图谱"""
    try:
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()

        # 如果有NLP结果，用它来构建
        if st.session_state["nlp_result"] is not None:
            nlp_result = st.session_state["nlp_result"]
            text = st.session_state["doc_result"].full_text
            kg.build_from_nlp_result(nlp_result, text)
        else:
            st.warning("建议先执行NLP分析以获得更丰富的知识图谱")
            return

        # 保存和可视化
        kg.save("data/processed/knowledge_graph.json")
        kg.visualize("data/processed/knowledge_graph.png")

        st.session_state["kg_built"] = True
        st.session_state["kg_stats"] = kg.get_stats()
        st.rerun()
    except Exception as e:
        st.error(f"知识图谱构建出错: {e}")


def _show_knowledge_graph():
    """显示知识图谱"""
    # 统计信息
    stats = st.session_state.get("kg_stats")
    if stats:
        cols = st.columns(3)
        with cols[0]:
            st.metric("节点数", stats.node_count)
        with cols[1]:
            st.metric("关系数", stats.edge_count)
        with cols[2]:
            st.metric("实体类型", len(stats.entity_types))

    # 图谱图片
    kg_path = "data/processed/knowledge_graph.png"
    if os.path.exists(kg_path):
        st.image(kg_path, caption="知识图谱可视化")
    else:
        st.write("图谱可视化未生成")


def render_rag_section():
    """渲染RAG问答区域"""
    if st.session_state["doc_result"] is None:
        st.info("👆 请先上传并解析文档")
        return

    st.subheader("💬 智能问答")

    # 初始化RAG引擎
    if not st.session_state["rag_ready"]:
        if st.button("🚀 初始化问答引擎", type="primary"):
            with st.spinner("正在初始化RAG引擎并导入文档..."):
                _init_rag()

    # 聊天界面
    if st.session_state["rag_ready"]:
        st.success("✅ 问答引擎已就绪，请提问！")

        # 显示聊天历史
        for msg in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant"):
                st.write(msg["answer"])
                if msg.get("sources"):
                    with st.expander("📚 查看来源"):
                        for src in msg["sources"]:
                            st.markdown(f"**第{src['page']}页**: {src['content']}")

        # 输入框
        question = st.chat_input("请输入你的问题...")
        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("正在检索并生成回答..."):
                    answer = _ask_question(question)

                st.write(answer.answer)
                if answer.sources:
                    with st.expander("📚 查看来源"):
                        for src in answer.sources:
                            st.markdown(f"**第{src['page']}页**: {src['content']}")

            # 记录历史
            st.session_state["chat_history"].append({
                "question": question,
                "answer": answer.answer,
                "sources": answer.sources,
            })


def _init_rag():
    """初始化RAG引擎"""
    try:
        from src.rag_engine import RAGEngine

        use_ollama = st.session_state.get("use_ollama", False)

        engine = RAGEngine(
            use_ollama=use_ollama,
            ollama_base_url=st.session_state.get("ollama_url", "http://localhost:11434"),
            ollama_model=st.session_state.get("ollama_model", "qwen2.5:7b"),
            ollama_embedding_model=st.session_state.get("ollama_embedding_model", "nomic-embed-text"),
            openai_model=st.session_state.get("openai_model", "gpt-4o-mini"),
            embedding_provider=st.session_state.get("embedding_provider", "ollama" if use_ollama else "openai"),
            hf_embedding_model=st.session_state.get("hf_embedding_model", "BAAI/bge-small-zh-v1.5"),
        )

        chunks = st.session_state.get("chunks", [])
        if chunks:
            engine.ingest_documents(chunks)

        st.session_state["rag_engine"] = engine
        st.session_state["rag_ready"] = True
        st.rerun()
    except Exception as e:
        st.error(f"RAG初始化出错: {e}")


def _ask_question(question: str):
    """提问"""
    from src.rag_engine import RAGAnswer

    engine = st.session_state.get("rag_engine")
    if engine is None:
        return RAGAnswer(
            question=question,
            answer="问答引擎未初始化",
        )

    top_k = st.session_state.get("top_k", 4)
    return engine.ask(question, top_k=top_k)


def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    render_header()

    # 主要内容区域
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 文档解析",
        "🧠 NLP分析",
        "🕸️ 知识图谱",
        "💬 智能问答",
    ])

    with tab1:
        render_upload_section()

    with tab2:
        render_nlp_section()

    with tab3:
        render_kg_section()

    with tab4:
        render_rag_section()


if __name__ == "__main__":
    main()
