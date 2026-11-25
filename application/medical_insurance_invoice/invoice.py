"""
@Project :   Study
@File    :   invoice.py
@Time    :   2026-11-25 14:42
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import argparse
import re
from pathlib import Path
import pymupdf  # uv add pymupdf

AMOUNT_PATTERNS = [
    # 优先匹配“金额合计 ... (小写)123.45”或“合计/总计”后的数字
    re.compile(
        r"(?:金额合计|费用合计|合计|总计|总金额)[\s\S]{0,120}?(?:\(小写\)|小写|金额)?[：:\s]*[¥￥]?\s*([0-9]+(?:,[0-9]{3})*\.[0-9]{2})"
    ),
    re.compile(
        r"[¥￥]\s*([0-9]+(?:,[0-9]{3})*\.[0-9]{2})\s*(?:合计|总计)"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="递归将 PDF 票据转为同目录 JPG，并在文件名后拼接合计金额")
    parser.add_argument("input_dir", type=Path, help="PDF 所在根路径")
    parser.add_argument("--dpi", type=int, default=250, help="渲染 DPI，默认 250")
    return parser.parse_args()


def extract_total_amount(doc: pymupdf.Document) -> str | None:
    text = "\n".join(page.get_text("text") for page in doc)
    for pattern in AMOUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_amount = match.group(1).replace(",", "")
            return f"{float(raw_amount):.2f}"
    return None


def convert_pdfs(root_dir: Path, dpi: int = 250) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是有效目录：{root_dir}")

    for pdf_path in sorted(root_dir.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue

        with pymupdf.open(pdf_path) as doc:
            amount = extract_total_amount(doc)
            amount_part = f"_{amount}" if amount else "_未识别金额"
            if not amount:
                print(f"[提示] 未提取到合计金额，使用占位标识：{pdf_path}")

            page_count = len(doc)
            for page_idx, page in enumerate(doc, start=1):
                page_part = "" if page_count == 1 else f"_p{page_idx}"
                out_name = f"{pdf_path.stem}{amount_part}{page_part}.jpg"
                out_path = pdf_path.parent / out_name
                pix = page.get_pixmap(dpi=dpi)
                pix.save(str(out_path))



# uv run python convert_pdfs.py ./pdfs
# uv run python convert_pdfs.py "/Users/name/报销/2026年11月" --dpi 300
if __name__ == "__main__":
    args = parse_args()
    convert_pdfs(args.input_dir, dpi=args.dpi)
