"""
知识图谱模块 (数据挖掘层)
===========================
从文档中提取实体关系，构建知识图谱，支持图谱分析与可视化。

技术栈:
    - NetworkX: 图结构存储与分析
    - Scikit-learn: 主题聚类 (KMeans)

使用示例:
    kg = KnowledgeGraphBuilder()
    kg.add_entities([("Google", "ORG"), ("AI", "TECH")])
    kg.add_relation("Google", "develops", "AI")
    kg.visualize("output/graph.png")
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx


@dataclass
class KGStats:
    """知识图谱统计信息"""
    node_count: int = 0
    edge_count: int = 0
    entity_types: dict = field(default_factory=dict)  # {type: count}
    top_nodes: list = field(default_factory=list)      # 度最高的节点

    def to_dict(self) -> dict:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
            "top_nodes": self.top_nodes,
        }


class KnowledgeGraphBuilder:
    """
    知识图谱构建器

    功能:
    1. 从实体列表构建图谱节点
    2. 添加实体间关系 (边)
    3. 从NLP结果自动提取关系
    4. 图谱分析 (中心性、社区检测)
    5. 可视化输出

    Args:
        graph_type: 图类型 ("directed" 或 "undirected")
    """

    def __init__(self, graph_type: str = "directed"):
        if graph_type == "directed":
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self._entity_types = {}  # {entity_text: entity_type}

    def add_entities(self, entities: list):
        """
        批量添加实体节点

        Args:
            entities: [(text, label), ...] 或 [Entity, ...]
        """
        for entity in entities:
            if hasattr(entity, "text"):
                text, label = entity.text, entity.label
            else:
                text, label = entity[0], entity[1]

            # 去重 (同一实体只添加一次)
            if text not in self.graph:
                self.graph.add_node(
                    text,
                    entity_type=label,
                    weight=1,
                )
                self._entity_types[text] = label
            else:
                # 已存在则增加权重
                self.graph.nodes[text]["weight"] += 1

    def add_relation(self, source: str, relation: str, target: str):
        """
        添加实体间的关系 (有向边)

        Args:
            source: 源实体
            relation: 关系类型
            target: 目标实体
        """
        # 确保节点存在
        if source not in self.graph:
            self.graph.add_node(source, entity_type="UNKNOWN")
        if target not in self.graph:
            self.graph.add_node(target, entity_type="UNKNOWN")

        self.graph.add_edge(source, target, relation=relation)

    def add_relations_from_text(self, text: str, entities: list):
        """
        从文本中自动提取实体间的共现关系

        简单的基于句子的共现分析:
        - 如果两个实体出现在同一句子中，建立"related_to"关系

        Args:
            text: 原文
            entities: 实体列表 [(text, label), ...]
        """
        entity_names = set()
        for e in entities:
            name = e.text if hasattr(e, "text") else e[0]
            entity_names.add(name)

        # 按句子分割
        sentences = re.split(r'[.。！？!?\n]', text)

        for sentence in sentences:
            # 找出本句中出现的实体
            present = [name for name in entity_names if name in sentence]

            # 两两建立共现关系
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    self.add_relation(present[i], "related_to", present[j])

    def build_from_nlp_result(self, nlp_result, text: str = ""):
        """
        从NLP分析结果构建知识图谱

        Args:
            nlp_result: NLPResult对象 (包含entities)
            text: 原文 (用于共现分析)
        """
        # 添加实体节点
        self.add_entities(nlp_result.entities)

        # 从文本中提取共现关系
        if text:
            self.add_relations_from_text(text, nlp_result.entities)

    def get_stats(self) -> KGStats:
        """获取图谱统计信息"""
        entity_type_counts = Counter(
            self.graph.nodes[n].get("entity_type", "UNKNOWN")
            for n in self.graph.nodes()
        )

        # 度最高的前10个节点
        degree_sorted = sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True
        )[:10]
        top_nodes = [
            {"node": n, "degree": d, "type": self.graph.nodes[n].get("entity_type", "?")}
            for n, d in degree_sorted
        ]

        return KGStats(
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
            entity_types=dict(entity_type_counts),
            top_nodes=top_nodes,
        )

    def find_paths(self, source: str, target: str, max_length: int = 5) -> list:
        """
        查找两个实体间的路径

        Args:
            source: 起始实体
            target: 目标实体
            max_length: 最大路径长度

        Returns:
            list of list: 路径列表
        """
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
            return paths
        except (nx.NetworkXError, nx.NodeNotFound):
            return []

    def get_subgraph(self, center_node: str, hops: int = 2) -> "KnowledgeGraphBuilder":
        """
        获取以某节点为中心的子图

        Args:
            center_node: 中心节点
            hops: 跳数

        Returns:
            KnowledgeGraphBuilder: 子图
        """
        if center_node not in self.graph:
            raise ValueError(f"节点不存在: {center_node}")

        # BFS获取邻居
        neighbors = set([center_node])
        frontier = set([center_node])
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            neighbors.update(next_frontier)
            frontier = next_frontier

        sub = KnowledgeGraphBuilder()
        sub.graph = self.graph.subgraph(neighbors).copy()
        return sub

    def visualize(
        self,
        output_path: str = "data/processed/knowledge_graph.png",
        figsize: tuple = (12, 8),
        max_nodes: int = 50,
    ) -> Optional[str]:
        """
        可视化知识图谱

        Args:
            output_path: 输出图片路径
            figsize: 图片尺寸
            max_nodes: 最多显示的节点数 (取度最高的)

        Returns:
            str: 输出文件路径，失败返回None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False

            # 如果节点太多，只展示度最高的
            graph = self.graph
            if graph.number_of_nodes() > max_nodes:
                top = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
                top_nodes = [n for n, _ in top]
                graph = graph.subgraph(top_nodes)

            fig, ax = plt.subplots(1, 1, figsize=figsize)

            # 布局
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

            # 节点颜色按实体类型
            color_map = {
                "PER": "#FF6B6B",
                "ORG": "#4ECDC4",
                "LOC": "#45B7D1",
                "DATE": "#96CEB4",
                "MISC": "#FFEAA7",
                "UNKNOWN": "#DDA0DD",
            }
            node_colors = [
                color_map.get(graph.nodes[n].get("entity_type", "UNKNOWN"), "#999999")
                for n in graph.nodes()
            ]

            # 节点大小按度
            node_sizes = [300 + graph.degree(n) * 100 for n in graph.nodes()]

            nx.draw_networkx_nodes(
                graph, pos, node_color=node_colors,
                node_size=node_sizes, alpha=0.8, ax=ax
            )
            nx.draw_networkx_labels(
                graph, pos, font_size=8, ax=ax
            )
            nx.draw_networkx_edges(
                graph, pos, alpha=0.3, arrows=True,
                arrowsize=10, ax=ax
            )

            ax.set_title(
                f"Knowledge Graph ({graph.number_of_nodes()} nodes, "
                f"{graph.number_of_edges()} edges)"
            )
            ax.axis("off")

            # 图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, label=label)
                for label, color in color_map.items()
            ]
            ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"[KG] 知识图谱已保存至: {output_path}")
            return output_path

        except Exception as e:
            print(f"[KG] 可视化出错: {e}")
            return None

    def save(self, output_path: str = "data/processed/knowledge_graph.json"):
        """保存图谱为JSON"""
        data = nx.node_link_data(self.graph)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[KG] 图谱数据已保存至: {output_path}")

    def load(self, input_path: str):
        """从JSON加载图谱"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        print(f"[KG] 图谱已加载: {self.graph.number_of_nodes()}节点, "
              f"{self.graph.number_of_edges()}边")


# ===== 主题聚类工具 =====

def cluster_documents(
    texts: list,
    n_clusters: int = 5,
) -> dict:
    """
    对文档列表进行主题聚类

    使用 TF-IDF + KMeans 进行文本聚类。

    Args:
        texts: 文档文本列表
        n_clusters: 聚类数量

    Returns:
        dict: {"labels": [...], "topics": [...], "top_words": [...]}
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    # KMeans聚类
    kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)

    # 提取每个簇的关键词
    feature_names = vectorizer.get_feature_names_out()
    top_words_per_cluster = []
    for cluster_id in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_words_per_cluster.append(top_words)

    return {
        "labels": labels.tolist(),
        "n_clusters": kmeans.n_clusters,
        "top_words": top_words_per_cluster,
    }


if __name__ == "__main__":
    # 快速测试
    kg = KnowledgeGraphBuilder()

    # 模拟添加实体
    entities = [
        ("Google", "ORG"), ("AI", "TECH"), ("DeepMind", "ORG"),
        ("AlphaGo", "PRODUCT"), ("Demis Hassabis", "PER"),
        ("London", "LOC"), ("Neural Network", "TECH"),
    ]
    kg.add_entities(entities)

    # 添加关系
    kg.add_relation("Google", "owns", "DeepMind")
    kg.add_relation("DeepMind", "created", "AlphaGo")
    kg.add_relation("Demis Hassabis", "leads", "DeepMind")
    kg.add_relation("DeepMind", "located_in", "London")
    kg.add_relation("AlphaGo", "uses", "Neural Network")

    # 统计
    stats = kg.get_stats()
    print(f"节点数: {stats.node_count}")
    print(f"边数: {stats.edge_count}")
    print(f"实体类型: {stats.entity_types}")
    print(f"Top节点: {stats.top_nodes}")

    # 查找路径
    paths = kg.find_paths("Google", "Neural Network")
    print(f"路径: {paths}")

    # 可视化
    kg.visualize("data/processed/test_kg.png")
