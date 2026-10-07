# GalaxiesML Dataset – Column Meaning Guide

The 127×127 validation HDF5 file contains 40,914 galaxies. Band-specific columns use g, r, i, z, and y for the five HSC photometric filters.

| Column | Unit / Shape | What it means |
|---|---|---|
| `coord` | deg | Coordinate information used for the HSC cone-search query (RA, DEC and search-radius related coordinate representation). |
| `dec` | degrees | Declination (J2000.0) of the image center / target position on the sky. |
| `g_central_image_pop_10px_rad` | — | Number of detected objects within a 10-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding around the target. |
| `g_central_image_pop_15px_rad` | — | Number of detected objects within a 15-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding on a larger scale. |
| `g_central_image_pop_5px_rad` | — | Number of detected objects within a 5-pixel-radius circle centered at the middle of the image. It is derived from the Source Extractor segmentation information and can indicate nearby-source crowding/contamination. |
| `g_cmodel_mag` | mag | CModel magnitude of the central galaxy in that photometric band. Magnitude measures apparent brightness; lower magnitude means a brighter apparent source. |
| `g_cmodel_magsigma` | mag | Uncertainty/error in the CModel magnitude measurement for the g-band. |
| `g_ellipticity` | — | Ellipticity of the galaxy in the g-band, defined by the dataset as 1 − B/A, where B is the semi-minor axis and A is the semi-major axis. Values closer to 0 indicate rounder shapes; larger values indicate more elongated shapes. |
| `g_half_light_radius` | pixels | Radius containing approximately 50% of the galaxy's total measured flux in the g-band. It is a measure of apparent galaxy size. |
| `g_isophotal_area` | pixels² | Area of the detected object inside the relevant isophotal/surface-brightness threshold in the g-band. |
| `g_major_axis` | pixels | Major axis of the detected galaxy/object in the g-band; the larger dimension of its measured shape. |
| `g_minor_axis` | pixels | Minor axis of the detected galaxy/object in the g-band; the smaller dimension of its measured shape. |
| `g_peak_surface_brightness` | mag/arcsec² | Peak surface brightness measured for the galaxy in the g-band. |
| `g_petro_rad` | pixels | Petrosian radius of the galaxy in the g-band, a light-profile-based measure of the object's apparent size. |
| `g_pos_angle` | degrees | Position angle/orientation of the galaxy's measured shape in the g-band. |
| `g_sersic_index` | — | Sérsic index describing the concentration/shape of the galaxy's light profile. Lower values are more disk-like; higher values are more centrally concentrated. |
| `i_central_image_pop_10px_rad` | — | Number of detected objects within a 10-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding around the target. |
| `i_central_image_pop_15px_rad` | — | Number of detected objects within a 15-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding on a larger scale. |
| `i_central_image_pop_5px_rad` | — | Number of detected objects within a 5-pixel-radius circle centered at the middle of the image. It is derived from the Source Extractor segmentation information and can indicate nearby-source crowding/contamination. |
| `i_cmodel_mag` | mag | CModel magnitude of the central galaxy in that photometric band. Magnitude measures apparent brightness; lower magnitude means a brighter apparent source. |
| `i_cmodel_magsigma` | mag | Uncertainty/error in the CModel magnitude measurement for the i-band. |
| `i_ellipticity` | — | Ellipticity of the galaxy in the i-band, defined by the dataset as 1 − B/A, where B is the semi-minor axis and A is the semi-major axis. Values closer to 0 indicate rounder shapes; larger values indicate more elongated shapes. |
| `i_half_light_radius` | pixels | Radius containing approximately 50% of the galaxy's total measured flux in the i-band. It is a measure of apparent galaxy size. |
| `i_isophotal_area` | pixels² | Area of the detected object inside the relevant isophotal/surface-brightness threshold in the i-band. |
| `i_major_axis` | pixels | Major axis of the detected galaxy/object in the i-band; the larger dimension of its measured shape. |
| `i_minor_axis` | pixels | Minor axis of the detected galaxy/object in the i-band; the smaller dimension of its measured shape. |
| `i_peak_surface_brightness` | mag/arcsec² | Peak surface brightness measured for the galaxy in the i-band. |
| `i_petro_rad` | pixels | Petrosian radius of the galaxy in the i-band, a light-profile-based measure of the object's apparent size. |
| `i_pos_angle` | degrees | Position angle/orientation of the galaxy's measured shape in the i-band. |
| `i_sersic_index` | — | Sérsic index describing the concentration/shape of the galaxy's light profile. Lower values are more disk-like; higher values are more centrally concentrated. |
| `image` | (N, 5, 127, 127) | The actual galaxy image data. Each galaxy has five 127×127 pixel images, one in each HSC band: g, r, i, z and y. Pixel values are stored as float32. |
| `object_id` | — | Unique HSC survey object identifier for the galaxy/object. |
| `r_central_image_pop_10px_rad` | — | Number of detected objects within a 10-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding around the target. |
| `r_central_image_pop_15px_rad` | — | Number of detected objects within a 15-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding on a larger scale. |
| `r_central_image_pop_5px_rad` | — | Number of detected objects within a 5-pixel-radius circle centered at the middle of the image. It is derived from the Source Extractor segmentation information and can indicate nearby-source crowding/contamination. |
| `r_cmodel_mag` | mag | CModel magnitude of the central galaxy in that photometric band. Magnitude measures apparent brightness; lower magnitude means a brighter apparent source. |
| `r_cmodel_magsigma` | mag | Uncertainty/error in the CModel magnitude measurement for the r-band. |
| `r_ellipticity` | — | Ellipticity of the galaxy in the r-band, defined by the dataset as 1 − B/A, where B is the semi-minor axis and A is the semi-major axis. Values closer to 0 indicate rounder shapes; larger values indicate more elongated shapes. |
| `r_half_light_radius` | pixels | Radius containing approximately 50% of the galaxy's total measured flux in the r-band. It is a measure of apparent galaxy size. |
| `r_isophotal_area` | pixels² | Area of the detected object inside the relevant isophotal/surface-brightness threshold in the r-band. |
| `r_major_axis` | pixels | Major axis of the detected galaxy/object in the r-band; the larger dimension of its measured shape. |
| `r_minor_axis` | pixels | Minor axis of the detected galaxy/object in the r-band; the smaller dimension of its measured shape. |
| `r_peak_surface_brightness` | mag/arcsec² | Peak surface brightness measured for the galaxy in the r-band. |
| `r_petro_rad` | pixels | Petrosian radius of the galaxy in the r-band, a light-profile-based measure of the object's apparent size. |
| `r_pos_angle` | degrees | Position angle/orientation of the galaxy's measured shape in the r-band. |
| `r_sersic_index` | — | Sérsic index describing the concentration/shape of the galaxy's light profile. Lower values are more disk-like; higher values are more centrally concentrated. |
| `ra` | degrees | Right Ascension (J2000.0) of the image center / target position on the sky. |
| `skymap_id` | — | Internal HSC sky-map position identifier, representing the survey tract/patch location. |
| `specz_dec` | degrees | Declination (J2000.0) of the galaxy in the spectroscopic survey. |
| `specz_flag_homogeneous` | Boolean | Homogenized spectroscopic-redshift quality flag. TRUE means the redshift is considered secure; FALSE means it is insecure. |
| `specz_mag_i` | mag | i-band magnitude of the galaxy in the spectroscopic survey. |
| `specz_name` | — | Name/identifier of the galaxy in the spectroscopic survey or surveys used for the spectroscopic redshift. |
| `specz_ra` | degrees | Right Ascension (J2000.0) of the galaxy in the spectroscopic survey. |
| `specz_redshift` | — | Spectroscopic redshift of the galaxy. This is the measured redshift used as the ground-truth redshift in many GalaxiesML applications. |
| `specz_redshift_err` | — | Uncertainty/error associated with the spectroscopic redshift. |
| `x_coord` | pixels | Image-plane X coordinate associated with the detected/central source in the galaxy cutout. |
| `x_coord_x` | 2 values | Auxiliary X-coordinate information stored as two values per galaxy; this is an image/source-coordinate field rather than a standard one-value metadata feature. |
| `x_coord_y` | 2 values | Auxiliary coordinate information paired with the X-coordinate field; stored as two values per galaxy. |
| `y_central_image_pop_10px_rad` | — | Number of detected objects within a 10-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding around the target. |
| `y_central_image_pop_15px_rad` | — | Number of detected objects within a 15-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding on a larger scale. |
| `y_central_image_pop_5px_rad` | — | Number of detected objects within a 5-pixel-radius circle centered at the middle of the image. It is derived from the Source Extractor segmentation information and can indicate nearby-source crowding/contamination. |
| `y_cmodel_mag` | mag | CModel magnitude of the central galaxy in that photometric band. Magnitude measures apparent brightness; lower magnitude means a brighter apparent source. |
| `y_cmodel_magsigma` | mag | Uncertainty/error in the CModel magnitude measurement for the y-band. |
| `y_coord` | pixels | Image-plane Y coordinate associated with the detected/central source in the galaxy cutout. |
| `y_coord_x` | 2 values | Auxiliary coordinate information paired with the Y-coordinate field; stored as two values per galaxy. |
| `y_coord_y` | 2 values | Auxiliary coordinate information paired with the Y-coordinate field; stored as two values per galaxy. |
| `y_ellipticity` | — | Ellipticity of the galaxy in the y-band, defined by the dataset as 1 − B/A, where B is the semi-minor axis and A is the semi-major axis. Values closer to 0 indicate rounder shapes; larger values indicate more elongated shapes. |
| `y_half_light_radius` | pixels | Radius containing approximately 50% of the galaxy's total measured flux in the y-band. It is a measure of apparent galaxy size. |
| `y_isophotal_area` | pixels² | Area of the detected object inside the relevant isophotal/surface-brightness threshold in the y-band. |
| `y_major_axis` | pixels | Major axis of the detected galaxy/object in the y-band; the larger dimension of its measured shape. |
| `y_minor_axis` | pixels | Minor axis of the detected galaxy/object in the y-band; the smaller dimension of its measured shape. |
| `y_peak_surface_brightness` | mag/arcsec² | Peak surface brightness measured for the galaxy in the y-band. |
| `y_petro_rad` | pixels | Petrosian radius of the galaxy in the y-band, a light-profile-based measure of the object's apparent size. |
| `y_pos_angle` | degrees | Position angle/orientation of the galaxy's measured shape in the y-band. |
| `y_sersic_index` | — | Sérsic index describing the concentration/shape of the galaxy's light profile. Lower values are more disk-like; higher values are more centrally concentrated. |
| `z_central_image_pop_10px_rad` | — | Number of detected objects within a 10-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding around the target. |
| `z_central_image_pop_15px_rad` | — | Number of detected objects within a 15-pixel-radius circle centered at the middle of the image. It measures nearby-source crowding on a larger scale. |
| `z_central_image_pop_5px_rad` | — | Number of detected objects within a 5-pixel-radius circle centered at the middle of the image. It is derived from the Source Extractor segmentation information and can indicate nearby-source crowding/contamination. |
| `z_cmodel_mag` | mag | CModel magnitude of the central galaxy in that photometric band. Magnitude measures apparent brightness; lower magnitude means a brighter apparent source. |
| `z_cmodel_magsigma` | mag | Uncertainty/error in the CModel magnitude measurement for the z-band. |
| `z_ellipticity` | — | Ellipticity of the galaxy in the z-band, defined by the dataset as 1 − B/A, where B is the semi-minor axis and A is the semi-major axis. Values closer to 0 indicate rounder shapes; larger values indicate more elongated shapes. |
| `z_half_light_radius` | pixels | Radius containing approximately 50% of the galaxy's total measured flux in the z-band. It is a measure of apparent galaxy size. |
| `z_isophotal_area` | pixels² | Area of the detected object inside the relevant isophotal/surface-brightness threshold in the z-band. |
| `z_major_axis` | pixels | Major axis of the detected galaxy/object in the z-band; the larger dimension of its measured shape. |
| `z_minor_axis` | pixels | Minor axis of the detected galaxy/object in the z-band; the smaller dimension of its measured shape. |
| `z_peak_surface_brightness` | mag/arcsec² | Peak surface brightness measured for the galaxy in the z-band. |
| `z_petro_rad` | pixels | Petrosian radius of the galaxy in the z-band, a light-profile-based measure of the object's apparent size. |
| `z_pos_angle` | degrees | Position angle/orientation of the galaxy's measured shape in the z-band. |
| `z_sersic_index` | — | Sérsic index describing the concentration/shape of the galaxy's light profile. Lower values are more disk-like; higher values are more centrally concentrated. |

**Note:** `x_coord_x`, `x_coord_y`, `y_coord_x`, and `y_coord_y` are auxiliary two-value coordinate arrays. The public GalaxiesML documentation defines `x_coord` and `y_coord` as image-plane coordinates but does not give a detailed public definition of the exact two-value encoding of these four auxiliary arrays.
