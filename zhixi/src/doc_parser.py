"""
文档解析模块 (CV层)
===================
负责从PDF文档中提取文本、表格和图像。

技术栈:
    - PyMuPDF (fitz): PDF文本和图像提取
    - pdfplumber: PDF表格提取
    - PaddleOCR: 图像中的文字识别 (降级方案)
    - OpenCV: 图像预处理

使用示例:
    parser = DocumentParser("path/to/document.pdf")
    result = parser.parse()
    print(result["text"])        # 提取的全文
    print(result["tables"])      # 提取的表格列表
    print(result["images"])      # 提取的图像路径列表
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from tqdm import tqdm


@dataclass
class PageContent:
    """单页解析结果"""
    page_number: int
    text: str = ""
    tables: list = field(default_factory=list)
    images: list = field(default_factory=list)  # 图像文件路径


@dataclass
class DocumentResult:
    """文档解析总结果"""
    filename: str
    total_pages: int
    pages: list = field(default_factory=list)  # List[PageContent]
    full_text: str = ""

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "full_text": self.full_text,
            "pages": [asdict(p) for p in self.pages],
        }

    def save(self, output_path: str):
        """保存解析结果为JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[DocParser] 解析结果已保存至: {output_path}")


class DocumentParser:
    """
    PDF文档解析器

    从PDF中提取:
    1. 文本内容 (PyMuPDF)
    2. 表格数据 (pdfplumber)
    3. 嵌入图像 (PyMuPDF)

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录 (存放提取的图像和JSON结果)
        extract_images: 是否提取图像，默认True
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "data/processed",
        extract_images: bool = True,
    ):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.extract_images = extract_images

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {self.pdf_path}")

        # 创建输出子目录
        self.doc_dir = self.output_dir / self.pdf_path.stem
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.doc_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

    def parse(self) -> DocumentResult:
        """
        执行完整的文档解析流程

        Returns:
            DocumentResult: 包含文本、表格、图像的解析结果
        """
        print(f"[DocParser] 开始解析: {self.pdf_path.name}")

        # 1. 用PyMuPDF提取文本和图像
        pages_text, pages_images = self._extract_with_pymupdf()

        # 2. 用pdfplumber提取表格
        pages_tables = self._extract_tables_with_pdfplumber()

        # 3. 组装结果
        total_pages = len(pages_text)
        pages = []
        all_text_parts = []

        for i in range(total_pages):
            page = PageContent(
                page_number=i + 1,
                text=pages_text[i] if i < len(pages_text) else "",
                tables=pages_tables[i] if i < len(pages_tables) else [],
                images=pages_images[i] if i < len(pages_images) else [],
            )
            pages.append(page)
            if page.text:
                all_text_parts.append(
                    f"--- 第{i + 1}页 ---\n{page.text}"
                )

        result = DocumentResult(
            filename=self.pdf_path.name,
            total_pages=total_pages,
            pages=pages,
            full_text="\n\n".join(all_text_parts),
        )

        print(
            f"[DocParser] 解析完成: {total_pages}页, "
            f"{len(result.full_text)}字符, "
            f"{sum(len(p.tables) for p in pages)}个表格, "
            f"{sum(len(p.images) for p in pages)}张图像"
        )
        return result

    def _extract_with_pymupdf(self) -> tuple:
        """使用PyMuPDF提取文本和图像"""
        doc = fitz.open(str(self.pdf_path))
        pages_text = []
        pages_images = []

        for page_num, page in enumerate(tqdm(doc, desc="提取文本和图像")):
            # 提取文本
            text = page.get_text("text").strip()
            pages_text.append(text)

            # 提取图像
            page_image_paths = []
            if self.extract_images:
                image_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        img_bytes = base_image["image"]
                        img_ext = base_image.get("ext", "png")
                        img_name = f"page{page_num + 1}_img{img_idx + 1}.{img_ext}"
                        img_path = self.images_dir / img_name
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        page_image_paths.append(str(img_path))

            pages_images.append(page_image_paths)

        doc.close()
        return pages_text, pages_images

    def _extract_tables_with_pdfplumber(self) -> list:
        """使用pdfplumber提取表格"""
        pages_tables = []

        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                for page in tqdm(pdf.pages, desc="提取表格"):
                    tables = page.extract_tables()
                    page_tables = []
                    for table in (tables or []):
                        if table:
                            # 将表格转换为字典格式
                            headers = table[0] if table else []
                            rows = table[1:] if len(table) > 1 else []
                            page_tables.append({
                                "headers": headers,
                                "rows": rows,
                                "row_count": len(rows),
                            })
                    pages_tables.append(page_tables)
        except Exception as e:
            print(f"[DocParser] 表格提取出错: {e}")
            # 返回空列表作为降级
            pages_tables = [[] for _ in range(self._get_page_count())]

        return pages_tables

    def _get_page_count(self) -> int:
        """获取PDF页数"""
        doc = fitz.open(str(self.pdf_path))
        count = len(doc)
        doc.close()
        return count

    def get_text_chunks(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list:
        """
        将文档文本切分为重叠的文本块 (用于RAG)

        Args:
            chunk_size: 每个块的字符数
            chunk_overlap: 相邻块的重叠字符数

        Returns:
            list of dict: [{"text": "...", "page": 1, "chunk_id": 0}, ...]
        """
        doc = self.parse()
        chunks = []
        chunk_id = 0

        for page in doc.pages:
            text = page.text
            if not text or len(text.strip()) < 10:
                continue

            # 按段落先分割
            paragraphs = text.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                # 如果加上这个段落会超过chunk_size
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page.page_number,
                        "chunk_id": chunk_id,
                    })
                    chunk_id += 1
                    # 保留重叠部分
                    current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk += ("\n\n" + para) if current_chunk else para

            # 处理剩余文本
            if current_chunk.strip():
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page.page_number,
                    "chunk_id": chunk_id,
                })
                chunk_id += 1

        print(f"[DocParser] 文本切分完成: {len(chunks)}个块")
        return chunks


# ===== 便捷函数 =====

def parse_pdf(
    pdf_path: str,
    output_dir: str = "data/processed",
    save_result: bool = True,
) -> DocumentResult:
    """
    一键解析PDF文档

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        save_result: 是否保存JSON结果

    Returns:
        DocumentResult: 解析结果
    """
    parser = DocumentParser(pdf_path, output_dir)
    result = parser.parse()

    if save_result:
        output_path = os.path.join(
            output_dir, Path(pdf_path).stem, "parse_result.json"
        )
        result.save(output_path)

    return result


if __name__ == "__main__":
    # 快速测试
    import sys

    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    else:
        print("用法: python doc_parser.py <pdf文件路径>")
        print("示例: python doc_parser.py data/sample_docs/test.pdf")
        sys.exit(0)

    result = parse_pdf(pdf_file)
    print(f"\n=== 文档摘要 ===")
    print(f"文件名: {result.filename}")
    print(f"页数: {result.total_pages}")
    print(f"总字符数: {len(result.full_text)}")
    print(f"\n=== 前500字 ===")
    print(result.full_text[:500])
