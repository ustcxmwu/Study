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

import pymupdf

AMOUNT_TOKEN_PATTERN = r"(?:[（(]\s*)?[+-−]?\s*[¥￥]?\s*[0-9]+(?:,[0-9]{3})*\.[0-9]{2}\s*(?:[）)])?"
AMOUNT_PATTERNS = [
    # 优先匹配“金额合计 ... (小写)-123.45”或“合计/总计”后的数字
    re.compile(
        rf"(?:金额合计|费用合计|合计|总计|总金额)[\s\S]{{0,120}}?(?:\(小写\)|小写|金额)?[：:\s]*({AMOUNT_TOKEN_PATTERN})"
    ),
    re.compile(
        rf"({AMOUNT_TOKEN_PATTERN})\s*(?:合计|总计)"
    ),
]


def extract_date(pdf_path: str) -> str | None:
    with pymupdf.open(pdf_path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    visit_match = re.search(r"就诊日期[:：]\s*(\d{8})", text)

    return visit_match.group(1) if visit_match else None


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
            raw_amount = match.group(1)
            normalized = (
                raw_amount.replace(" ", "")
                .replace(",", "")
                .replace("¥", "")
                .replace("￥", "")
                .replace("−", "-")
                .replace("（", "(")
                .replace("）", ")")
            )
            if normalized.startswith("(") and normalized.endswith(")"):
                normalized = f"-{normalized[1:-1]}"
            return f"{float(normalized):.2f}"
    return None


def convert_pdfs(root_dir: Path, dpi: int = 250) -> None:
    if not root_dir.exists() or not root_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是有效目录：{root_dir}")

    for pdf_path in sorted(root_dir.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue
        if "收费明细" in pdf_path.name:
            print(f"[跳过] 收费明细 PDF，跳过处理：{pdf_path}")
            continue

        with pymupdf.open(pdf_path) as doc:
            visit_date = extract_date(pdf_path)
            date_part = f"{visit_date}" if visit_date else "_未识别日期"
            if not visit_date:
                print(f"[提示] 未提取到就诊日期，使用占位标识：{pdf_path}")
            amount = extract_total_amount(doc)
            amount_part = f"_{amount}" if amount else "_未识别金额"
            if not amount:
                print(f"[提示] 未提取到合计金额，使用占位标识：{pdf_path}")

            base_name = f"{date_part}{amount_part}"
            date_dir = root_dir / visit_date if visit_date else pdf_path.parent
            if visit_date:
                date_dir.mkdir(parents=True, exist_ok=True)

            page_count = len(doc)
            for page_idx, page in enumerate(doc, start=1):
                page_part = "" if page_count == 1 else f"_p{page_idx}"
                out_name = f"{base_name}{page_part}.jpg"
                out_path = date_dir / out_name
                pix = page.get_pixmap(dpi=dpi)
                pix.save(str(out_path))

        renamed_pdf_path = date_dir / f"{base_name}.pdf"
        if pdf_path != renamed_pdf_path:
            if renamed_pdf_path.exists():
                print(f"[提示] PDF 目标文件已存在，跳过重命名：{renamed_pdf_path}")
            else:
                pdf_path.rename(renamed_pdf_path)
                print(f"[重命名] {pdf_path.name} -> {renamed_pdf_path.name}")


# uv run python convert_pdfs.py ./pdfs
# uv run python convert_pdfs.py "/Users/name/报销/2026年11月" --dpi 300
if __name__ == "__main__":
    args = parse_args()
    convert_pdfs(args.input_dir, dpi=args.dpi)
