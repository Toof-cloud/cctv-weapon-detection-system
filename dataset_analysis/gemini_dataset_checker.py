import argparse
import csv
import os
from collections import Counter
from pathlib import Path

from PIL import Image
from google import genai


PROMPT = """
You are reviewing a weapon detection dataset.

Classify the image into EXACTLY ONE category.

VALID HANDGUN:
- Semi-automatic Pistol
- Revolver
- Compact Pistol
- Other Handgun Variant

VALID KNIFE:
- Kitchen Knife
- Utility Knife
- Folding Knife
- Combat/Hunting Knife
- Chef Knife
- Other Knife Variant

INVALID:
- Toy Gun
- LEGO Gun
- Prop Gun
- Water Gun
- Painting / Artwork
- Poster
- Magazine Cover
- Statue / Display Model
- Cartoon / Drawing
- Invalid Annotation
- Wrong Weapon Type

MANUAL REVIEW:
- Handle Only
- Partial Weapon
- Extremely Blurry
- Ambiguous Object

Return ONLY:

CATEGORY: <category>
STATUS: <VALID|INVALID|MANUAL_REVIEW>
REASON: <short reason>
"""


def build_client():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Export it before running this script."
        )

    return genai.Client(api_key=api_key)


def collect_images(folder_path):
    folder = Path(folder_path)
    images = []

    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(folder.glob(pattern))

    return sorted(images)


def write_results(csv_name, results):
    with open(csv_name, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_id", "category", "status", "reason"])

        for row in results:
            writer.writerow(
                [row["image_id"], row["category"], row["status"], row["reason"]]
            )

    invalid_csv = f"invalid_{csv_name}"
    with open(invalid_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        for row in results:
            if row["status"] == "INVALID":
                writer.writerow([row["image_id"]])


def audit_folder(folder_path, csv_name, client):
    results = []
    images = collect_images(folder_path)[:1]

    print(f"Processing {len(images)} images from {folder_path}")

    if not images:
        print("No images found. Check the folder path and file extensions.")

    for img_path in images:
        image_id = img_path.stem

        try:
            image = Image.open(img_path)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[PROMPT, image],
            )

            text = (response.text or "").strip()
            category = ""
            status = ""
            reason = ""

            for line in text.splitlines():
                if line.startswith("CATEGORY:"):
                    category = line.replace("CATEGORY:", "", 1).strip()
                elif line.startswith("STATUS:"):
                    status = line.replace("STATUS:", "", 1).strip()
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "", 1).strip()

            results.append(
                {
                    "image_id": image_id,
                    "category": category,
                    "status": status,
                    "reason": reason,
                }
            )

            print(image_id, status)

        except Exception as exc:
            print("\n===================")
            print("IMAGE:", image_id)
            print("ERROR TYPE:", type(exc).__name__)
            print("ERROR:", exc)
            print("===================\n")

            results.append(
                {
                    "image_id": image_id,
                    "category": "ERROR",
                    "status": "MANUAL_REVIEW",
                    "reason": str(exc),
                }
            )

    write_results(csv_name, results)

    print("\nDone")
    if results:
        print(dict(Counter(row["status"] for row in results)))
    else:
        print("No results were written.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Review a folder of weapon dataset images with Gemini."
    )
    parser.add_argument(
        "folder_path",
        nargs="?",
        default=r"review_batch\reviewer_b\handguns",
        help="Folder containing images to review.",
    )
    parser.add_argument(
        "csv_name",
        nargs="?",
        default="reviewer_b_handguns.csv",
        help="Output CSV file name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    client = build_client()
    audit_folder(args.folder_path, args.csv_name, client)


if __name__ == "__main__":
    main()