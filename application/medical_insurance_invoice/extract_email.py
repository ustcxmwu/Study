"""
@Project :   Study
@File    :   extract_email.py
@Time    :   2026-11-25 16:47
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
from pathlib import Path
from email import policy
from email.parser import BytesParser


def get_unique_path(base_dir: Path, raw_filename: str) -> Path:
    file_name = Path(raw_filename).name
    candidate = base_dir / file_name
    if not candidate.exists():
        return candidate

    suffixes = "".join(Path(file_name).suffixes)
    stem = file_name[: -len(suffixes)] if suffixes else file_name
    index = 1
    while True:
        next_name = f"{stem}_{index}{suffixes}"
        candidate = base_dir / next_name
        if not candidate.exists():
            return candidate
        index += 1


def extract_attachments(eml_dir: str, out_dir: str) -> None:
    src = Path(eml_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for eml_path in src.rglob("*.eml"):
        with eml_path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        saved_count = 0

        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                if filename and payload is not None:
                    target = get_unique_path(dst, filename)
                    target.write_bytes(payload)
                    saved_count += 1

        if saved_count:
            print(f"{eml_path.name}: 提取 {saved_count} 个附件 -> {dst}")


if __name__ == "__main__":
    extract_attachments("/Users/wuxiaomin/@个人/敢敢和在在/email", "/Users/wuxiaomin/@个人/敢敢和在在/附件")