import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("data/GalaxyML/5x127x127_training_with_morphology.csv")

# 84911, 17072, 17146

# sersic_index, petro_rad, pos_angle are not correlated enough to be sent into a common average

drop_and_average = [
    "central_image_pop_10px_rad",
    "central_image_pop_15px_rad",
    "isophotal_area",
    "cmodel_mag",
    "central_image_pop_5px_rad",
    "half_light_radius",
    "ellipticity",
    "minor_axis",
    "major_axis",
    # "sersic_index", "petro_rad", "pos_angle" # Too Uncorrelated
]

drop_with_bands = ["peak_surface_brightness", "cmodel_magsigma"]
drop_without_bands = [
    "skymap_id",
    # "object_id",
    "specz_name",
    "specz_ra",
    "specz_dec",
    "x_coord",
    "y_coord",
    "specz_flag_homogeneous",
    "average_central_image_pop_10px_rad",
    "ra",
    "dec",
    "average_cmodel_mag",  # High Corr with specz_mag_i
    "coord",
]


bands_columns_to_remove = [
    f"{band}_{column}"
    for band in "giryz"
    for column in drop_and_average + drop_with_bands
]

band_averages = [
    [f"{band}_{column}" for band in "giryz"] for column in drop_and_average
]


for i, column in enumerate(drop_and_average):
    data[f"average_{column}"] = data[band_averages[i]].mean(axis=1)

data = data.drop(columns=bands_columns_to_remove)
data = data.drop(columns=drop_without_bands)

CORR = data.corr(numeric_only=True).abs()
CORR = CORR[CORR < 1]
# CORR = CORR[CORR >= 0.8][CORR < 1]

CORR = CORR.dropna(axis=1, how="all")

CORR.to_csv("CORRHIGHNA.csv")

CORR.describe().T

sns.heatmap(
    CORR[CORR > 0.5].fillna(0),
    annot=True,
    fmt=".2f",
)
plt.show()


data["average_major_axis"].hist()
data[data["average_major_axis"] > 20]
