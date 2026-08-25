from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt

# pos = coords.SkyCoord(ra=229.525576, dec=42.7458538, unit="deg", frame="icrs")
pos = coords.SkyCoord.from_name("NGC 4826")
xid = SDSS.query_region(pos, radius="3 arcmin")
print(xid)

bands = ["u", "g", "r", "i", "z"]
images = {band: SDSS.get_images(matches=xid, band=band) for band in bands}
print(images)


for i in range(len(images["u"])):
    fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
    fig.suptitle(f"SDSS Images - Object {i}", fontsize=16)
    for idx, b in enumerate(bands):
        image_data = images[b][i][0].data.copy()
        image_data[image_data > 1] = 1
        image_data[image_data < -1] = -1
        axes[idx].imshow(image_data, cmap='gray')  # Optional: add colormap
        axes[idx].set_title(f"Band {b}")
        axes[idx].axis('off')  # Remove axis ticks for cleaner look
    plt.tight_layout()
    plt.show()
