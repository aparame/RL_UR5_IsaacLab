import json
import cv2
import os
import argparse
import random
import numpy as np


def verify(dataset_dir):
    meta_path = os.path.join(dataset_dir, "metadata.jsonl")
    if not os.path.exists(meta_path):
        print(f"Metadata not found at {meta_path}")
        return

    entries = []
    with open(meta_path, "r") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except:
                pass

    if not entries:
        print("No entries found.")
        return

    print(f"Found {len(entries)} entries.")

    # Pick random samples
    samples = random.sample(entries, min(5, len(entries)))

    output_dir = "verification_output"
    os.makedirs(output_dir, exist_ok=True)

    for idx, sample in enumerate(samples):
        img_path = sample["image_path"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(dataset_dir, img_path)

        if not os.path.exists(img_path):
            # Try finding relative to metadata
            # Assume image_path is absolute or relative properly?
            # In code we saved absolute path to key, but maybe json has it differently?
            # Code: metadata[meta_key] where meta_key = abspath
            # But we added v["image_path"] = k
            pass

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue

        # Draw BBox
        bbox = sample["bbox"]  # x1, y1, x2, y2
        x1, y1, x2, y2 = map(int, bbox)

        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Text
        cv2.putText(
            img,
            f"Env {sample['env']} Step {sample['step']}",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

        out_path = os.path.join(output_dir, f"verify_{idx}.png")
        cv2.imwrite(out_path, img)
        print(f"Saved {out_path} with BBox {bbox}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="agan_dataset")
    args = parser.parse_args()
    verify(args.dataset)
