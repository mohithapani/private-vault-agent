"""
Standalone OCR diagnostic script.

Bypasses indexing/chunking entirely so you can see exactly what PaddleOCR
extracts from a single PDF page (or a standalone image), with and without
the crop_scanned_document step, side by side. Useful for figuring out
whether a "missing"/partial field (like an address) is an OCR problem or
something further down the pipeline.

Usage:
    # A specific page of a PDF (1-indexed)
    python debug_ocr.py --pdf "streamlit_workspace/Mohit_Passport.pdf" --page 2

    # A standalone image file
    python debug_ocr.py --image path/to/photo.jpg

    # Also save the cropped image to disk so you can open and eyeball it
    python debug_ocr.py --pdf file.pdf --page 2 --save-crop cropped_page2.png

    # Try a higher render DPI (more detail, slower)
    python debug_ocr.py --pdf file.pdf --page 2 --dpi 400
"""

import argparse
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from main import run_ocr_on_image, crop_scanned_document, ocr


def show(label: str, text: str) -> None:
    print(f"\n===== {label} ({len(text)} chars) =====")
    print(text if text.strip() else "[EMPTY]")


def raw_full_res_ocr(pil_image: Image.Image) -> str:
    """Runs PaddleOCR directly on the image with NO thumbnail/resize step at
    all -- unlike run_ocr_on_image, which always caps to a max of 2500x2500
    before OCR. Use this to check whether that cap is degrading small text
    (e.g. apartment/unit numbers) on a large rendered page.
    """
    img_array = np.array(pil_image.convert("RGB"))
    result = ocr.predict(img_array)
    lines = []
    for block in result:
        for line in block["rec_texts"]:
            lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", help="Path to a PDF file")
    parser.add_argument("--page", type=int, default=1, help="1-indexed page number (used with --pdf)")
    parser.add_argument("--image", help="Path to a standalone image file instead of a PDF")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for PDF pages (default: 300, same as main.py)")
    parser.add_argument("--save-crop", help="Optional path to save the cropped image for visual inspection")
    parser.add_argument("--raw", action="store_true",
                         help="Also run OCR on the full-resolution image with NO 2500px resize cap, to compare")
    args = parser.parse_args()

    if args.pdf:
        pages = convert_from_path(args.pdf, dpi=args.dpi, first_page=args.page, last_page=args.page)
        pil_image = pages[0]
        print(f"Rendered page {args.page} of {args.pdf} at {args.dpi} DPI, size={pil_image.size}")
    elif args.image:
        pil_image = Image.open(args.image)
        pil_image.load()
        print(f"Loaded {args.image}, size={pil_image.size}")
    else:
        raise SystemExit("Provide --pdf (with --page) or --image")

    cropped = crop_scanned_document(pil_image)
    print(f"Cropped size={cropped.size}")

    # Mirror the exact resize run_ocr_on_image applies internally, just to
    # show what size actually reaches PaddleOCR in the normal pipeline.
    thumbnailed = cropped.convert("RGB").copy()
    thumbnailed.thumbnail((2500, 2500), Image.Resampling.LANCZOS)
    print(f"Size actually sent to OCR by the normal pipeline (post 2500px cap): {thumbnailed.size}")

    if args.save_crop:
        cropped.save(args.save_crop)
        print(f"Saved cropped image to {args.save_crop} -- "
              f"open it and check nothing important (like the address block) got cut off.")

    text_with_crop = run_ocr_on_image(pil_image, crop=True)
    text_no_crop = run_ocr_on_image(pil_image, crop=False)

    show("WITH crop (what indexing actually uses)", text_with_crop)
    show("WITHOUT crop", text_no_crop)

    if text_no_crop.strip() and not text_with_crop.strip():
        print("\n⚠️  Cropping removed ALL text that OCR could otherwise find -- crop is very likely the problem here.")
    elif len(text_no_crop) > len(text_with_crop) * 1.3:
        print("\n⚠️  Uncropped OCR found noticeably more text than cropped -- crop may be clipping part of the page.")

    if args.raw:
        text_raw = raw_full_res_ocr(cropped)
        show(f"RAW full-resolution OCR, no 2500px cap (size={cropped.convert('RGB').size})", text_raw)
        if len(text_raw) > len(text_with_crop) * 1.15 or text_raw != text_with_crop:
            print("\n⚠️  Full-resolution OCR differs from the capped/thumbnailed version -- "
                  "the 2500px resize is very likely degrading recognition on small text.")


if __name__ == "__main__":
    main()