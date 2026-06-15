# 智析 (ZhiXi) - 多模态文档智能分析与知识问答平台

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 项目简介

**智析 (ZhiXi)** 是一个端到端的多模态文档智能分析与知识问答平台。它能够从 PDF 等文档中提取文本、表格和图像，通过 NLP 技术进行深度分析，构建知识图谱，并提供基于 RAG（检索增强生成）的智能问答功能。

## ✨ 核心功能

- **📄 文档解析**: 支持 PDF 文本、表格、图像提取，集成 OCR 能力
- **🔍 NLP 分析**: 实体识别、关键词提取、文本摘要
- **🕸️ 知识图谱**: 自动构建和分析领域知识图谱
- **💬 RAG 问答**: 基于检索增强生成的智能问答引擎
- **🌐 Web 界面**: 基于 Streamlit 的交互式用户界面

## 🏗️ 项目结构

```
.
├── notebooks/              # Jupyter 实验笔记本
│   ├── 01_numpy_pandas_exercise.ipynb
│   ├── 02_ocr_demo.ipynb
│   ├── 03_rag_demo.ipynb
│   ├── 04_nlp_demo.ipynb
│   ├── 05_knowledge_graph_demo.ipynb
│   └── 06_streamlit_demo.ipynb
├── src/                    # 核心源代码
│   ├── doc_parser.py       # 文档解析模块
│   ├── nlp_pipeline.py     # NLP 分析流水线
│   ├── knowledge_graph.py  # 知识图谱构建与分析
│   ├── rag_engine.py       # RAG 检索增强问答引擎
│   └── app.py              # Streamlit Web 应用
├── tests/                  # 单元测试
│   ├── test_doc_parser.py
│   ├── test_nlp_pipeline.py
│   ├── test_knowledge_graph.py
│   └── test_rag_engine.py
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐在虚拟环境中运行

### 安装步骤

1. **克隆仓库**
```bash
git clone <your-repo-url>
cd zhi-xi
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **启动 Web 应用**
```bash
streamlit run src/app.py
```

## 📚 模块说明

### 1. 文档解析 (`src/doc_parser.py`)
- 使用 PyMuPDF 提取 PDF 文本和元数据
- 使用 PaddleOCR 进行光学字符识别
- 支持表格检测和提取

### 2. NLP 流水线 (`src/nlp_pipeline.py`)
- 命名实体识别 (NER)
- 关键词提取
- 自动文本摘要

### 3. 知识图谱 (`src/knowledge_graph.py`)
- 基于 NetworkX 构建知识图谱
- 实体关系抽取
- 图分析和可视化

### 4. RAG 引擎 (`src/rag_engine.py`)
- 使用 LangChain 框架
- ChromaDB 向量数据库
- 支持语义检索和问答

### 5. Web 应用 (`src/app.py`)
- Streamlit 交互式界面
- 文档上传与解析
- 实时问答演示

## 🧪 测试

运行所有测试：
```bash
pytest tests/ -v
```

## 📓 示例 Notebook

项目包含 6 个 Jupyter Notebook，用于演示各模块功能：

| Notebook | 描述 |
|----------|------|
| `01_numpy_pandas_exercise.ipynb` | NumPy/Pandas 基础练习 |
| `02_ocr_demo.ipynb` | OCR 文字识别演示 |
| `03_rag_demo.ipynb` | RAG 问答系统演示 |
| `04_nlp_demo.ipynb` | NLP 处理流程演示 |
| `05_knowledge_graph_demo.ipynb` | 知识图谱构建演示 |
| `06_streamlit_demo.ipynb` | Streamlit 应用演示 |

## 🛠️ 技术栈

- **文档处理**: PyMuPDF, PaddleOCR
- **NLP**: Transformers, spaCy, NLTK
- **LLM/RAG**: LangChain, ChromaDB
- **知识图谱**: NetworkX
- **Web 框架**: Streamlit
- **数据处理**: NumPy, Pandas
- **测试**: pytest

## 📝 使用示例

### Python API 调用

```python
from src.doc_parser import DocumentParser
from src.nlp_pipeline import NLPPipeline
from src.rag_engine import RAGEngine

# 解析文档
parser = DocumentParser()
doc = parser.parse("example.pdf")

# NLP 分析
nlp = NLPPipeline()
entities = nlp.extract_entities(doc.text)
keywords = nlp.extract_keywords(doc.text)

# RAG 问答
rag = RAGEngine()
rag.index_documents([doc])
answer = rag.query("文档的主要内容是什么？")
print(answer)
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过 Issue 联系我们。

---

**Made with ❤️ by ZhiXi Team**
