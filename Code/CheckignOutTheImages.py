import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

training = h5py.File("data/GalaxyML/5x64x64_training_with_morphology.hdf5", "r")
data = pd.read_csv("data/GalaxyML/preprocessedMetadata.csv")


# %%


# %%
target_id = 74643924559360902
ids = training["object_id"][:]


def getImage(target_id):
    matches = np.where(ids == target_id)[0]
    current_image = training["image"][matches].squeeze(0)
    return current_image


def showImage(current_image, title=None):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(current_image[i], cmap="gray")
        axes[i].set_title("grizy"[i])
        axes[i].axis("off")
    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
    else:
        plt.tight_layout()
    plt.show()

def show_rgb(image, title=None, Q=8, stretch=0.5):
    g, r, i = image[0], image[1], image[2]
    rgb = np.stack([i, r, g], axis=-1).astype(float)
    # Per-channel percentile clip first
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], [1, 99])
        rgb[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-8), 0, None)
    # Arcsinh stretch
    rgb = np.arcsinh(stretch * Q * rgb) / np.arcsinh(stretch * Q)
    rgb = np.clip(rgb, 0, 1)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

showImage(getImage(target_id))


target_id = int(
    data[data["major_minor_axis_ratio"] == data["major_minor_axis_ratio"].min()][
        "object_id"
    ].iloc[0]
)


# %%


N = 100  # number of samples

# Sort by major_minor_axis_ratio so "even gaps" spans the full range
sorted_df = data.sort_values("major_minor_axis_ratio").reset_index(drop=True)

base = 100
positions = list(np.linspace(base, len(sorted_df) - 1, N).astype(int))

items = sorted_df.loc[positions]
target_ids = list(items["object_id"])
images = [getImage(i) for i in target_ids]
# titles = [
#     " ".join(f"{k}={v}" for k, v in item._asdict().items())
#     for item in items.itertuples()
# ]


for i in range(len(images)):
    showImage(images[i], title=target_ids[i])


for i in range(len(images)):
    show_rgb(images[i], title=target_ids[i])

