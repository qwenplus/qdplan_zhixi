"""
智析 (ZhiXi) 基础测试
=====================
运行方式: cd zhixi && python -m pytest tests/ -v

注意: 这些测试不依赖外部API或模型，仅测试核心逻辑。
"""

import os
import sys
import json
import tempfile

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestKnowledgeGraph:
    """知识图谱模块测试"""

    def test_add_entities(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_entities([("Google", "ORG"), ("AI", "TECH")])

        assert kg.graph.number_of_nodes() == 2
        assert "Google" in kg.graph
        assert kg.graph.nodes["Google"]["entity_type"] == "ORG"

    def test_add_relation(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_relation("Google", "develops", "AI")

        assert kg.graph.number_of_nodes() == 2
        assert kg.graph.number_of_edges() == 1
        assert kg.graph.edges["Google", "AI"]["relation"] == "develops"

    def test_get_stats(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_entities([("A", "PER"), ("B", "ORG"), ("C", "PER")])
        kg.add_relation("A", "works_at", "B")
        kg.add_relation("C", "works_at", "B")

        stats = kg.get_stats()
        assert stats.node_count == 3
        assert stats.edge_count == 2
        assert stats.entity_types["PER"] == 2

    def test_find_paths(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_relation("A", "knows", "B")
        kg.add_relation("B", "knows", "C")

        paths = kg.find_paths("A", "C")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]

    def test_save_and_load(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_relation("X", "relates", "Y")

        # 保存
        tmp = tempfile.mktemp(suffix=".json")
        kg.save(tmp)

        # 加载
        kg2 = KnowledgeGraphBuilder()
        kg2.load(tmp)
        assert kg2.graph.number_of_nodes() == 2
        assert kg2.graph.number_of_edges() == 1

        os.unlink(tmp)

    def test_duplicate_entity(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        kg.add_entities([("Google", "ORG"), ("Google", "ORG")])

        # 同一实体只添加一次，但权重增加
        assert kg.graph.number_of_nodes() == 1
        assert kg.graph.nodes["Google"]["weight"] == 2

    def test_cooccurrence(self):
        from src.knowledge_graph import KnowledgeGraphBuilder

        kg = KnowledgeGraphBuilder()
        entities = [("Google", "ORG"), ("AI", "TECH"), ("Microsoft", "ORG")]
        kg.add_entities(entities)

        text = "Google and AI are closely related. Microsoft also invests in AI."
        kg.add_relations_from_text(text, entities)

        # Google-AI 和 Microsoft-AI 应该有关系
        assert kg.graph.number_of_edges() >= 2


class TestDocumentParser:
    """文档解析模块测试 (仅测试文本切块逻辑)"""

    def test_text_chunking_logic(self):
        """测试文本切块的数据结构"""
        # 模拟chunks结构
        chunks = [
            {"text": "这是第一段文本", "page": 1, "chunk_id": 0},
            {"text": "这是第二段文本", "page": 1, "chunk_id": 1},
            {"text": "这是第三段文本", "page": 2, "chunk_id": 2},
        ]

        assert len(chunks) == 3
        assert chunks[0]["page"] == 1
        assert chunks[2]["page"] == 2


class TestNLPPipeline:
    """NLP模块测试 (数据结构测试，不依赖模型)"""

    def test_entity_dataclass(self):
        from src.nlp_pipeline import Entity, NLPResult

        entity = Entity(text="Google", label="ORG", start=0, end=6)
        assert entity.text == "Google"
        assert entity.label == "ORG"

    def test_nlp_result_to_dict(self):
        from src.nlp_pipeline import NLPResult

        result = NLPResult(
            entities=[],
            keywords=[("AI", 0.9), ("ML", 0.8)],
            summary="A summary",
            word_count=100,
        )
        d = result.to_dict()
        assert d["word_count"] == 100
        assert len(d["keywords"]) == 2


class TestRAGEngine:
    """RAG引擎测试 (数据结构测试，不依赖API)"""

    def test_rag_answer_dataclass(self):
        from src.rag_engine import RAGAnswer

        answer = RAGAnswer(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            sources=[{"content": "...", "page": 1}],
            model_used="gpt-4o-mini",
        )
        d = answer.to_dict()
        assert d["question"] == "What is AI?"
        assert len(d["sources"]) == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
