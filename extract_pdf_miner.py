"""Extract text from a PDF using pdfminer.six.

Usage:
    python extract_pdf_miner.py input.pdf [-o output.txt]
"""
import argparse
import logging
import os
import sys
from typing import Optional, Sequence

from pdfminer.high_level import extract_text

logger = logging.getLogger("extract_pdf_miner")


def extract_text_miner(pdf_path: str, output_path: str) -> bool:
    """Write the text of ``pdf_path`` to ``output_path``. Returns True on success."""
    if not os.path.exists(pdf_path):
        logger.error("PDF file not found at %s", pdf_path)
        return False
    try:
        text = extract_text(pdf_path)
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(text)
    except Exception as exc:  # pdfminer raises several unrelated error types
        logger.error("Could not extract text: %s", exc)
        return False
    logger.info("Successfully extracted text to %s", output_path)
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract text from a PDF with pdfminer.six")
    parser.add_argument("pdf", help="path to the input PDF")
    parser.add_argument("-o", "--output", default="pdf_content_miner.txt", help="output text file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return 0 if extract_text_miner(args.pdf, args.output) else 1


if __name__ == "__main__":
    sys.exit(main())
