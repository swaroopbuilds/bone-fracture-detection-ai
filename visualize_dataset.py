import os
import matplotlib.pyplot as plt

# Folder where Flask saves classified images
classified_dir = "classified"

# Count number of classified images
def count_classified_images(path):
    counts = {}
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            counts[folder] = len(os.listdir(folder_path))
    return counts

counts = count_classified_images(classified_dir)
print("Classified Image Counts:", counts)

# --- Visualization ---
plt.figure(figsize=(6,5))
plt.bar(counts.keys(), counts.values(), color=['lightcoral', 'skyblue'])
plt.title("Model-Classified Image Distribution")
plt.xlabel("Prediction Category")
plt.ylabel("Number of Images")
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save visualization in visuals folder
os.makedirs("visuals", exist_ok=True)
plt.tight_layout()
plt.savefig("visuals/classified_output_distribution.png")
plt.show()

print("\n✅ Visualization saved in 'visuals/classified_output_distribution.png'")
