# 智析 (ZhiXi) - 多模态文档智能分析与知识问答平台

<div align="center">

📊 **上传 PDF → 自动解析 → NLP 分析 → 构建知识图谱 → 智能问答**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)

</div>

---

## 📖 项目简介

**智析 (ZhiXi)** 是一个端到端的多模态文档智能分析平台，集成了文档解析、NLP 分析、知识图谱构建和 RAG 智能问答功能。用户上传 PDF 文档后，系统可自动提取文本/表格/图像，进行命名实体识别、关键词提取和摘要生成，构建可视化知识图谱，并支持基于文档内容的智能问答。

### ✨ 核心特性

- **📄 多模态文档解析**：从 PDF 中提取文本、表格和图像（支持 OCR）
- **🧠 NLP 智能分析**：命名实体识别、关键词提取、自动摘要、词云可视化
- **🕸️ 知识图谱**：自动抽取实体关系，构建并可视化知识图谱
- **💬 RAG 问答**：基于检索增强生成的智能问答，支持溯源
- **🎨 Web 界面**：开箱即用的 Streamlit 交互界面

---

## 🏗️ 项目结构

```
zhixi/
├── src/                      # 核心源代码
│   ├── __init__.py
│   ├── doc_parser.py         # 文档解析模块 (PDF→文本/表格/图像)
│   ├── nlp_pipeline.py       # NLP 分析模块 (NER/关键词/摘要)
│   ├── knowledge_graph.py    # 知识图谱构建与分析
│   ├── rag_engine.py         # RAG 问答引擎
│   └── app.py                # Streamlit Web 应用
├── notebooks/                # Jupyter 实验笔记本
│   ├── 01_numpy_pandas_basics.ipynb
│   ├── 02_ocr_experiment.ipynb
│   ├── 03_rag_experiment.ipynb
│   ├── 04_nlp_analysis.ipynb
│   ├── 05_knowledge_graph.ipynb
│   └── 06_streamlit_app.ipynb
├── tests/                    # 单元测试
│   ├── __init__.py
│   └── test_core.py
├── requirements.txt          # 依赖配置
└── README.md                 # 项目文档
```

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.9+
- (可选) Ollama 本地 LLM 服务
- (可选) OpenAI API Key

### 2. 安装依赖

```bash
cd zhixi
pip install -r requirements.txt
```

> 💡 **提示**: 首次运行会自动下载 NLP 模型（约几百 MB），请确保网络畅通。

### 3. 配置环境变量（可选）

创建 `.env` 文件配置 API Key：

```bash
# OpenAI API (使用 OpenAI 模型时必需)
OPENAI_API_KEY=sk-your-api-key-here
CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama 配置 (使用本地模型时)
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
HF_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 4. 启动 Web 应用

```bash
streamlit run src/app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

---

## 📦 模块详解

### 1. 文档解析模块 (`doc_parser.py`)

**功能**: 从 PDF 中提取文本、表格和图像

```python
from src.doc_parser import DocumentParser

parser = DocumentParser("document.pdf")
result = parser.parse()

print(result.full_text)      # 全文
print(result.pages[0].tables)  # 第一页表格
print(result.pages[0].images)  # 第一页图像路径

# 切分文本块 (用于 RAG)
chunks = parser.get_text_chunks(chunk_size=500, chunk_overlap=50)
```

**技术栈**: PyMuPDF, pdfplumber, PaddleOCR, OpenCV

---

### 2. NLP 分析模块 (`nlp_pipeline.py`)

**功能**: 命名实体识别、关键词提取、自动摘要

```python
from src.nlp_pipeline import NLPPipeline

nlp = NLPPipeline()
result = nlp.analyze(text)

print(result.entities)    # [(实体文本，类型), ...]
print(result.keywords)    # [(关键词，分数), ...]
print(result.summary)     # 摘要文本

# 生成词云
nlp.generate_wordcloud(text, "output/wordcloud.png")
```

**技术栈**: Transformers, KeyBERT, spaCy, WordCloud

---

### 3. 知识图谱模块 (`knowledge_graph.py`)

**功能**: 从实体关系构建知识图谱，支持分析与可视化

```python
from src.knowledge_graph import KnowledgeGraphBuilder

kg = KnowledgeGraphBuilder()
kg.add_entities([("Google", "ORG"), ("AI", "TECH")])
kg.add_relation("Google", "develops", "AI")

# 获取统计
stats = kg.get_stats()
print(f"节点数：{stats.node_count}, 边数：{stats.edge_count}")

# 查找路径
paths = kg.find_paths("Google", "AI")

# 可视化
kg.visualize("output/graph.png")
```

**技术栈**: NetworkX, Scikit-learn

---

### 4. RAG 问答引擎 (`rag_engine.py`)

**功能**: 基于文档的智能问答，支持溯源

```python
from src.rag_engine import RAGEngine

engine = RAGEngine(use_ollama=False)  # 或 use_ollama=True 使用本地模型
engine.ingest_documents(chunks)       # 导入文档块

answer = engine.ask("这篇论文的主要贡献是什么？")
print(answer.answer)
print(answer.sources)  # 来源文档块
```

**支持的 Embedding 方案**:
- **Ollama** (本地，免费): `nomic-embed-text`
- **HuggingFace** (中文推荐): `BAAI/bge-small-zh-v1.5`
- **OpenAI** (付费): `text-embedding-3-small`

**技术栈**: LangChain, ChromaDB, OpenAI/Ollama

---

## 🎨 Web 界面功能

启动应用后，你将看到四个主要标签页：

| 标签页 | 功能 |
|--------|------|
| 📄 **文档解析** | 上传 PDF，查看提取的文本/表格/图像统计 |
| 🧠 **NLP 分析** | 执行实体识别、关键词提取、摘要生成、词云可视化 |
| 🕸️ **知识图谱** | 构建并可视化知识图谱，查看实体关系 |
| 💬 **智能问答** | 基于文档内容进行对话式问答，支持溯源 |

### 侧边栏配置

- **LLM 模式**: 选择本地 Ollama 或 OpenAI API
- **Embedding 模型**: 选择向量化方案（Ollama/HuggingFace/OpenAI）
- **RAG 参数**: 调整文本块大小、重叠量、检索数量

---

## 🧪 运行测试

```bash
cd zhixi
python -m pytest tests/ -v
```

测试覆盖核心数据结构和不依赖外部 API 的逻辑。

---

## 📓 Jupyter Notebook 教程

`notebooks/` 目录包含 6 个交互式教程：

| Notebook | 内容 |
|----------|------|
| `01_numpy_pandas_basics.ipynb` | NumPy/Pandas 基础 |
| `02_ocr_experiment.ipynb` | OCR 文字识别实验 |
| `03_rag_experiment.ipynb` | RAG 问答实验 |
| `04_nlp_analysis.ipynb` | NLP 分析实战 |
| `05_knowledge_graph.ipynb` | 知识图谱构建 |
| `06_streamlit_app.ipynb` | Streamlit 应用演示 |

---

## 🔧 高级用法

### 命令行快速问答

```bash
python src/rag_engine.py document.pdf "你的问题"
```

### 使用本地 Ollama 模型

1. 安装 Ollama: https://ollama.ai
2. 拉取模型:
   ```bash
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
3. 在 Web 界面选择"本地 Ollama"模式

### 批量处理文档

```python
from src.doc_parser import parse_pdf
from src.rag_engine import RAGEngine

# 批量解析
docs = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
all_chunks = []

for doc in docs:
    parser = DocumentParser(doc)
    chunks = parser.get_text_chunks()
    all_chunks.extend(chunks)

# 统一导入 RAG
engine = RAGEngine()
engine.ingest_documents(all_chunks)
```

---

## 📋 依赖说明

| 类别 | 主要库 | 用途 |
|------|--------|------|
| 基础科学 | numpy, pandas, scikit-learn | 数据处理与机器学习 |
| 文档解析 | PyMuPDF, pdfplumber, PaddleOCR | PDF 文本/表格/图像提取 |
| NLP | transformers, keybert, spacy | 实体识别/关键词/摘要 |
| LLM/RAG | langchain, chromadb, openai | 向量检索与问答 |
| 知识图谱 | networkx | 图结构构建与分析 |
| Web | streamlit | 交互式界面 |
| 可视化 | matplotlib, wordcloud | 图表与词云 |

完整依赖见 [`requirements.txt`](requirements.txt)。

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢以下开源项目：

- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [NetworkX](https://github.com/networkx/networkx)

---

<div align="center">

**智析 (ZhiXi)** © 2024 | Made with ❤️ for Document Intelligence

</div>
