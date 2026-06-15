"""
NLP分析模块
=============
提供命名实体识别(NER)、关键词提取、自动摘要、情感分析等功能。

技术栈:
    - HuggingFace Transformers: NER、摘要生成
    - KeyBERT: 关键词提取
    - wordcloud: 词云可视化

使用示例:
    nlp = NLPPipeline()
    result = nlp.analyze("一段长文本...")
    print(result["entities"])    # 实体列表
    print(result["keywords"])    # 关键词列表
    print(result["summary"])     # 摘要
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Entity:
    """命名实体"""
    text: str
    label: str       # PER(人名), ORG(组织), LOC(地点), DATE(日期) 等
    start: int = 0
    end: int = 0


@dataclass
class NLPResult:
    """NLP分析结果"""
    entities: list = field(default_factory=list)       # List[Entity]
    keywords: list = field(default_factory=list)       # List[(keyword, score)]
    summary: str = ""
    word_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class NLPPipeline:
    """
    NLP分析管道

    集成多个NLP工具，提供一站式文本分析能力:
    1. 命名实体识别 (NER) - 识别人名、组织、地点等
    2. 关键词提取 - 基于KeyBERT的语义关键词
    3. 自动摘要 - 基于Transformer的抽取式/生成式摘要
    4. 词云生成 - 文本关键词可视化

    Args:
        ner_model: NER模型名称，默认使用多语言模型
        summary_model: 摘要模型名称
        device: 运行设备 ("cpu" 或 "cuda")
    """

    def __init__(
        self,
        ner_model: str = "dslim/bert-base-NER",
        summary_model: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
    ):
        self.ner_model_name = ner_model
        self.summary_model_name = summary_model
        self.device = device

        # 延迟加载模型 (按需初始化，节省内存)
        self._ner_pipeline = None
        self._summary_pipeline = None
        self._keybert_model = None

    def _load_ner(self):
        """加载NER模型"""
        if self._ner_pipeline is None:
            from transformers import pipeline
            self._ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
                device=self.device,
            )
            print("[NLP] NER模型加载完成")

    def _load_summary(self):
        """加载摘要模型"""
        if self._summary_pipeline is None:
            from transformers import pipeline
            self._summary_pipeline = pipeline(
                "summarization",
                model=self.summary_model_name,
                device=self.device,
            )
            print("[NLP] 摘要模型加载完成")

    def _load_keybert(self):
        """加载KeyBERT"""
        if self._keybert_model is None:
            from keybert import KeyBERT
            self._keybert_model = KeyBERT()
            print("[NLP] KeyBERT加载完成")

    def analyze(
        self,
        text: str,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        generate_summary: bool = True,
        top_k_keywords: int = 10,
    ) -> NLPResult:
        """
        对文本执行完整的NLP分析

        Args:
            text: 输入文本
            extract_entities: 是否提取实体
            extract_keywords: 是否提取关键词
            generate_summary: 是否生成摘要
            top_k_keywords: 关键词数量

        Returns:
            NLPResult: 包含实体、关键词、摘要的分析结果
        """
        if not text or len(text.strip()) < 10:
            print("[NLP] 文本过短，跳过分析")
            return NLPResult()

        result = NLPResult(word_count=len(text.split()))

        # 1. 命名实体识别
        if extract_entities:
            result.entities = self.extract_entities(text)

        # 2. 关键词提取
        if extract_keywords:
            result.keywords = self.extract_keywords(text, top_k=top_k_keywords)

        # 3. 自动摘要
        if generate_summary:
            result.summary = self.generate_summary(text)

        return result

    def extract_entities(self, text: str) -> list:
        """
        命名实体识别 (NER)

        识别人名(PER)、组织(ORG)、地点(LOC)等实体。

        Args:
            text: 输入文本

        Returns:
            list of Entity: 识别到的实体列表
        """
        self._load_ner()

        try:
            raw_entities = self._ner_pipeline(text[:4096])  # 限制长度避免OOM
            entities = []
            for ent in raw_entities:
                entities.append(Entity(
                    text=ent["word"],
                    label=ent["entity_group"],
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                ))
            print(f"[NLP] 识别到 {len(entities)} 个实体")
            return entities
        except Exception as e:
            print(f"[NLP] NER出错: {e}")
            return []

    def extract_keywords(self, text: str, top_k: int = 10) -> list:
        """
        关键词提取 (基于KeyBERT)

        使用语义相似度提取最具代表性的关键词/短语。

        Args:
            text: 输入文本
            top_k: 返回的关键词数量

        Returns:
            list of tuple: [(keyword, score), ...] 按分数降序
        """
        self._load_keybert()

        try:
            keywords = self._keybert_model.extract_keywords(
                text[:4096],
                keyphrase_ngram_range=(1, 2),  # 1-2个词的短语
                stop_words="english",
                top_n=top_k,
            )
            print(f"[NLP] 提取到 {len(keywords)} 个关键词")
            return keywords
        except Exception as e:
            print(f"[NLP] 关键词提取出错: {e}")
            return []

    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """
        自动摘要生成

        Args:
            text: 输入文本 (英文文本效果最佳)
            max_length: 摘要最大长度(词数)

        Returns:
            str: 生成的摘要文本
        """
        self._load_summary()

        try:
            # BART模型输入限制约1024 tokens
            input_text = text[:3000]
            result = self._summary_pipeline(
                input_text,
                max_length=max_length,
                min_length=30,
                do_sample=False,
            )
            summary = result[0]["summary_text"]
            print(f"[NLP] 摘要生成完成 ({len(summary)}字符)")
            return summary
        except Exception as e:
            print(f"[NLP] 摘要生成出错: {e}")
            # 降级: 返回前200字
            return text[:200] + "..."

    def generate_wordcloud(self, text: str, output_path: str = "data/processed/wordcloud.png"):
        """
        生成词云图

        Args:
            text: 输入文本
            output_path: 输出图片路径
        """
        try:
            from wordcloud import WordCloud

            # 去除停用词和短词
            cleaned = re.sub(r'\b\w{1,2}\b', '', text)

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                colormap="viridis",
            ).generate(cleaned)

            wc.to_file(output_path)
            print(f"[NLP] 词云已保存至: {output_path}")
            return output_path
        except Exception as e:
            print(f"[NLP] 词云生成出错: {e}")
            return None


# ===== 便捷函数 =====

def analyze_text(
    text: str,
    entities: bool = True,
    keywords: bool = True,
    summary: bool = True,
) -> NLPResult:
    """
    一键分析文本

    Args:
        text: 输入文本
        entities: 是否提取实体
        keywords: 是否提取关键词
        summary: 是否生成摘要

    Returns:
        NLPResult: 分析结果
    """
    pipeline = NLPPipeline()
    return pipeline.analyze(
        text,
        extract_entities=entities,
        extract_keywords=keywords,
        generate_summary=summary,
    )


if __name__ == "__main__":
    # 快速测试
    sample_text = """
    Artificial intelligence (AI) is transforming industries worldwide.
    Companies like Google, Microsoft, and OpenAI are leading the development
    of large language models. In 2024, the global AI market was valued at
    over $200 billion. Researchers at Stanford University and MIT have
    published groundbreaking papers on transformer architectures and
    reinforcement learning from human feedback (RLHF).
    """

    print("=== NLP分析测试 ===\n")
    result = analyze_text(sample_text)

    print(f"词数: {result.word_count}")
    print(f"\n实体: {[(e.text, e.label) for e in result.entities]}")
    print(f"关键词: {result.keywords[:5]}")
    print(f"摘要: {result.summary}")
