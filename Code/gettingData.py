# https://pypi.org/project/sdss/
# We can also wget it wget --spider https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/2505/3/frame-r-002505-3-0038.fits.bz2
# Documentation at https://www.sdss4.org/dr17/data_access/bulk/wget --spider https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/platelist.fits

from sdss import Region

ra = 179.689293428354
dec = 0.454379056007667

reg = Region(ra, dec, fov=0.033)
reg.show()

# %% 
# needs a data folder in pwd
# from sdss.photometry import frame_filename, obj_frame_url, \
#      download_file, unzip, get_df, df_radec2pixel
#
# objid = 1237646587710014999
#
# zip_file = 'data/' + frame_filename(objid) + '.fits.bz2'
# fits_file = zip_file[:-4]
# jpg_file = fits_file.replace('-r-', '-irg-').replace('fits', 'jpg')
#
# zip_url = obj_frame_url(objid, 'r')
# download_file(zip_url, 'data/')
# unzip(zip_file)
#
# jpg_url = obj_frame_url(objid, 'irg', jpg=True)
# download_file(jpg_url, 'data/')
#
# df = get_df(objid)
# df = df_radec2pixel(df=df, fits_file=fits_file)
#
# df.to_csv('data/COMP.csv', index=False)
#
