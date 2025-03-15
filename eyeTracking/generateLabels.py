import os

# Define dataset paths
image_folder = "dataset/images/train"
label_folder = "dataset/labels/train"
os.makedirs(label_folder, exist_ok=True)

# Generate labels
for img_name in os.listdir(image_folder):
    if img_name.endswith((".jpg", ".png")):
        label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.8 0.8\n")
            f.write("1 0.5 0.5 0.3 0.3\n")

print("Labels generated successfully!")
