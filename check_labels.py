import cv2
import glob
import os
import matplotlib.pyplot as plt

def verify_sample(image_dir, label_dir, count=3):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.*")))[:count]
    class_names = {0: "Crop", 1: "Weed"}
    colors = {0: (0, 200, 0), 1: (220, 0, 0)}  # Green: Crop, Red: Weed

    fig, axes = plt.subplots(1, count, figsize=(15, 5))
    if count == 1:
        axes = [axes]

    for idx, img_path in enumerate(img_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{base}.txt")

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls, xc, yc, bw, bh = map(float, parts)
                    cls = int(cls)

                    # Denormalize YOLO coordinates to bounding box pixels
                    xmin = int((xc - bw / 2) * w)
                    ymin = int((yc - bh / 2) * h)
                    xmax = int((xc + bw / 2) * w)
                    ymax = int((yc + bh / 2) * h)

                    color = colors.get(cls, (255, 255, 255))
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(
                        img, class_names.get(cls, str(cls)),
                        (xmin, max(15, ymin - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                    )

        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(base, fontsize=9)
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_sample("dataset/train/images", "dataset/train/labels", count=3)