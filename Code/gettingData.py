from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt

radec = "227.376696 +42.6534463"
ra, dec = map(float, radec.split())
pos = coords.SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

# pos = coords.SkyCoord.from_name("NGC 4826")
xid = SDSS.query_region(pos, radius="60 arcsec")
xid = xid[:2]
print(xid)


bands = ["u", "g", "r", "i", "z"]
images = {}
for band in bands:
    images[band] = SDSS.get_images(matches=xid, band=band)


print(images)




for i in range(len(images["u"])):
    fig, axes = plt.subplots(1, len(bands), figsize=(15, 5))
    fig.suptitle(f"SDSS Images - Object {i}", fontsize=16)
    for idx, b in enumerate(bands):
        image_data = images[b][i][0].data.copy()
        image_data[image_data > 1] = 1
        image_data[image_data < -1] = -1
        axes[idx].imshow(image_data, cmap="gray")  # Optional: add colormap
        axes[idx].set_title(f"Band {b}")
        axes[idx].axis("off")  # Remove axis ticks for cleaner look
    plt.tight_layout()
    plt.savefig("UHHHH.png")
    plt.show()

import numpy as np

def normalize(data):
    data1 = data.copy()
    data1[data1 > 1] = 1
    return data1
    return (data - np.min(data)) / (np.max(data) - np.min(data))


rgbImg = np.stack([normalize(images[b][0][0].data) for b in ["u", "r", "i"]], axis = -1)
plt.imshow(rgbImg)
plt.axis('off')
plt.savefig('rgb_image.png', dpi=300, bbox_inches='tight')
plt.show()
