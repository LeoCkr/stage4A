# Import every library needed
#%%
import matplotlib
from LakesModeling_Subprogram import *
from geopy.distance import geodesic
from rasterio.plot import show
from matplotlib import gridspec
#%% md



# Init every parameter
#%%
# name of the lake to model
# available names : Amadeus, Austin, Barlee, Blanche, Carey, Carnegie, Cowan, Disappointment, Everard, Eyre_North, Eyre_South, Frome, Gairdner, Gregory, Island_Lagoon, Lefroy, Mackay, Macleod, Moore, Torrens, Yamma_Yamma
lake_to_model = 'Amadeus'

# path to folder containing data (shapefiles, icesatdata, tiff files, etc.)
data_folder = 'path/to/data/folder' + lake_to_model

# path to icesatdata containing all topography data
icesatdata_path = data_folder + '/' + lake_to_model + '_icesatdata'

# path to shapefile containing geometry of each lake
lakes_shapefile = 'path/to/shapefile'

# define the time period of icesatdata
time_period = ['2021-01-01', '2022-01-01']

# define the pixel size (in degree per pixel)
pixel_size = 0.005

# define the region size around the lake (geological context)
region_size = 0.1

# define the clipping distance to the lake edges
distance_to_edges = 0.01

# define the filter coefficient (if alpha = 1 then 1% of minimal values and 1% of maximal values will be removed)
alpha = 1

# path to the shapefile containing rivers of Australia
rivers_shapefile = 'path/to/rivers/shapefile'

# path to hydroSHEDS tiff file
hydro_file = data_folder + '/hydro.tif'

# CRS of 2D projection
projection_crs = 'EPSG:7853'
#%% md



# Open and select the lake geodataframe
#%%
# file containing all lakes geometry
lakes = geopandas.read_file(lakes_shapefile)

# select the lake to model
lake = lakes[lakes.Lake_Name == lake_to_model]

# Obtain the bounds of the lake
lake_bounds = lake.bounds.iloc[0]
#%% md



# Get icesatadata (ONLY if you did not download it already)
#%%
download_icesat_data(spatial_bounds=[lake_bounds],
                     time_period=time_period,
                     destination=icesatdata_path)
#%% md



# extract data and rasterize it
#%%
# extract elevation points from corresponding hdf5 files
original_geospatial_points = extract_data_from_folder(folder_path=icesatdata_path)

# rasterize elevation points with a given pixel size
rasterize_geospatial_data(data=original_geospatial_points,
                          output_path=data_folder,
                          pixel_size=pixel_size)
#%% md



# Clip data with the lake shape, get the confluence points and plot everything
#%%
# ignore warnings due to CRS
warnings.filterwarnings("ignore")

# open tiff file corresponding to icesat elevation data
with rasterio.open(data_folder + '/original_raster.tif') as file:
    # clip the data within the lake
    inside_raster, inside_transform = clip_geospatial_data_with_mask(geospatial_data=file,
                                                                     mask_gdf=lake,
                                                                     buffer_distance=0.005,
                                                                     clip_type='within',
                                                                     region_scaling=0.6)

with rasterio.open(data_folder + '/temp.tif',
                   'w',
                   driver='GTiff',
                   height=inside_raster.shape[0],
                   width=inside_raster.shape[1],
                   count=1, dtype=inside_raster.dtype,
                   crs='EPSG:4326',
                   transform=inside_transform) as dst:
    dst.write(interpolate_missing_data(inside_raster), 1)

with rasterio.open(data_folder + '/temp.tif') as file:
    # clip the data within the lake
    interpolated_inside_raster, interpolated_inside_transform = clip_geospatial_data_with_mask(geospatial_data=file,
                                                                                               mask_gdf=lake,
                                                                                               buffer_distance=0.005,
                                                                                               clip_type='within',
                                                                                               region_scaling=0.6)

# open tiff file corresponding to hydroSHEDS elevation data
with rasterio.open(hydro_file)as file:
    # clip the data without the lake
    outside_raster, outside_transform = clip_geospatial_data_with_mask(geospatial_data=file,
                                                                       mask_gdf=lake,
                                                                       buffer_distance=0,
                                                                       clip_type='without',
                                                                       region_scaling=0.6)

# extract edges as linestring
confluence = extract_confluence_points(lake_geodataframe=lake,
                                       path_to_rivers_shapefile=rivers_shapefile)
#%% md



# Plot the rivers (if needed)
#%%
rivers = geopandas.read_file(rivers_shapefile)
#%%
river = rivers[rivers['geometry'].apply(lambda x: x.intersects(lake.iloc[0].geometry))]
river = river.to_crs('EPSG:4326')
xmin, ymin, xmax, ymax = rivers.total_bounds
large_polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
inverse_polygons = geopandas.GeoSeries([large_polygon]).difference(lake.unary_union)

clipped_lines_outside = geopandas.clip(river, inverse_polygons)
#%%
# plot the result
fig, ax = matplotlib.pyplot.subplots(figsize=(20,20))
norm = matplotlib.colors.TwoSlopeNorm(vmin=numpy.nanpercentile(outside_raster, 2), vcenter=numpy.nanpercentile(outside_raster, 30), vmax=numpy.nanpercentile(outside_raster, 90))

show(outside_raster, transform=outside_transform, ax=ax, cmap='binary', norm=norm)
show(inside_raster, transform=inside_transform, ax=ax, cmap='gist_earth')
ax.scatter(confluence[0], confluence[1], c='red', marker='2', s=15)
for i in range(len(confluence[1])):
    ax.annotate(str(i), xy=(confluence[0][i],confluence[1][i]), xytext=(confluence[0][i],confluence[1][i]+0.01), size=10, ha='center')
clipped_lines_outside.plot(ax=ax)
#%% md



# Filter main confluence points
#%%
selected_indexes = [7,10,40,0]
main_confluence = [confluence[0][selected_indexes], confluence[1][selected_indexes]]

# plot the result
fig, ax = matplotlib.pyplot.subplots(figsize=(20,20))
c = show(inside_raster, transform=inside_transform, ax=ax, cmap='gist_earth')
show(outside_raster, transform=outside_transform, ax=ax, cmap='binary')
ax.scatter(main_confluence[0], main_confluence[1], c='red')
ax.set_xlabel('Longitude (m)',fontsize=20)
ax.set_ylabel('Latitude (m)',fontsize=20)
cbar = matplotlib.pyplot.colorbar(c1, ax=ax)
cbar.set_label('Altitude (m)')

ax.set_title('Eyre North Elevations',fontsize=20)
fig.savefig('./report5.png', dpi=100)
#%% md



# Modeling
#%%
# convert the raster into geospatial points
points = unrasterize(raster_data=inside_raster,
                     geo_transform=inside_transform)

# filter out the points considered as noise
filtered_points = filter_extreme_altitudes(geospatial_data=points,
                                           filter_threshold=alpha)

# project the points to have everything in meters
projected_filtered_points = transform_coordinates(points_data=[filtered_points[0],filtered_points[1]],
                                                  crs=projection_crs)
projected_filtered_points.append(filtered_points[2])

# model the points to obtain the plane parameters (we fit the lake to a plane)
model = model_altitude_from_coordinates(geospatial_data=projected_filtered_points)

# extract edges points
edges_points = extract_geospatial_edges(geometry=lake.geometry.iloc[0])

# project edges also
projected_edges_points = transform_coordinates(points_data=edges_points,
                                               crs=projection_crs)

# use the previous parameters to plot edges within the plane
ztemp = []
for i in range(len(projected_edges_points[0])):
    ztemp.append(projected_edges_points[0][i]*model[0] + projected_edges_points[1][i]*model[1] + model[2])
projected_edges_points.append(numpy.array(ztemp))

# remove longitude and latitude offset
corrected_projected_edges_points = projected_edges_points.copy()
corrected_projected_filtered_points = projected_filtered_points.copy()
corrected_projected_edges_points[:2] = [projected_edges_points[i] - numpy.min(projected_filtered_points[i]) for i in range(2)]
corrected_projected_filtered_points[:2] = [projected_filtered_points[i] - numpy.min(projected_filtered_points[i]) for i in range(2)]

corrected_model = model_altitude_from_coordinates(geospatial_data=corrected_projected_filtered_points)
#%% md



# plot result
#%%
# plot result
fig = matplotlib.pyplot.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')
ax.set_title('Modeling')

ax.plot(corrected_projected_edges_points[0],corrected_projected_edges_points[1],corrected_projected_edges_points[2], c='black')
ax.scatter(corrected_projected_filtered_points[0], corrected_projected_filtered_points[1], corrected_projected_filtered_points[2], c=corrected_projected_filtered_points[2], cmap='gist_earth')

# Plot the plane
plane_colors = matplotlib.pyplot.cm.inferno((corrected_model[5] - numpy.min(corrected_model[5])) / (numpy.max(corrected_model[5]) - numpy.min(corrected_model[5])))
ax.plot_surface(corrected_model[3], corrected_model[4], corrected_model[5], facecolors=plane_colors, alpha=0.1)

ax.set_xlabel('Longitude (in meters)')
ax.set_ylabel('Latitude (in meters)')
ax.set_zlabel('Altitude (in meters)')

# ax.view_init(elev=90, azim=-90)  # elev=90 to see it from above

matplotlib.pyplot.show()
#%% md



# Polar projection
#%%
# get indexes of the points with min and max altitude value
i_min = numpy.where(projected_edges_points[2] == numpy.min(projected_edges_points[2]))[0][0]
i_max = numpy.where(projected_edges_points[2] == numpy.max(projected_edges_points[2]))[0][0]

# calculate distance between the two points
point_max = (edges_points[0][i_max], edges_points[1][i_max])
point_min = (edges_points[0][i_min], edges_points[1][i_min])
distance = geodesic((point_max[1],point_max[0]), (point_min[1],point_min[0])).kilometers
azimuth = round(calculate_initial_compass_bearing(pointA=point_max, pointB=point_min),2)

# elevation difference
alt_diff = projected_edges_points[2][i_max] - projected_edges_points[2][i_min]

# get the slope
slope = round(alt_diff / distance, 3)

centroid_lon = (point_max[0] + point_min[0]) / 2
centroid_lat = (point_max[1] + point_min[1]) / 2

edges_r, edges_theta = convert_to_polar_coordinates(data=edges_points,
                                                    reference_point=[centroid_lon, centroid_lat])

point_max_r, point_max_theta = convert_to_polar_coordinates(data=point_max,
                                                            reference_point=[centroid_lon, centroid_lat])

point_min_r, point_min_theta = convert_to_polar_coordinates(data=point_min,
                                                            reference_point=[centroid_lon, centroid_lat])

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(20,20))
c = ax.scatter(edges_theta, edges_r, c=projected_edges_points[2], cmap='gist_earth', s=2)
ax.plot([point_max_theta, point_min_theta], [point_max_r, point_min_r], c='red', lw=5, linestyle='dashed')
plt.colorbar(c, ax=ax, label='Altitude')
ax.set_title("Projection polaire du lac")
ax.set_xticklabels(['E\n0°', 'N-E\n45°', 'N\n90°', 'N-W\n135°', 'W\n180°', 'S-W\n225°', 'S\n270°', 'S-E\n315°'])

plt.show()
#%% md



# Final plot
#%%
df = generate_lake_information(lake_gdf=lake,
                               time_period=time_period,
                               resolution=pixel_size,
                               filter_threshold=alpha,
                               target_crs=projection_crs,
                               r_squared_value=model[6]*100,
                               slope=slope,
                               azimuth=azimuth)

fig = matplotlib.pyplot.figure(figsize=(21, 29.7))
gs0 = gridspec.GridSpec(3, 2, figure=fig)

face_color = (255/255, 255/255, 255/255)
labels_color = 'black'
fig.set_facecolor(face_color)

ax0 = fig.add_subplot(gs0[0, :])
ax0.set_facecolor(face_color)
# Masquer les axes
ax0.axis('off')
ax0.axis('tight')
# Affichage du tableau
table = ax0.table(cellText=df.values, colLabels=df.columns, cellLoc = 'left', loc='center')
ax0.set_title(lake_to_model, color=labels_color, fontsize=32)
table.scale(0.9, 2)
for key, cell in table.get_celld().items():
    row, col = key
    if row == 0:  # Pour les headers
        cell.set_fontsize(22)
        cell.set_facecolor('grey')
        cell.set_text_props(color='white')
    else:
        cell.set_fontsize(16)
        cell.set_facecolor('darkgrey')
        cell.set_text_props(color='white')

ax1 = fig.add_subplot(gs0[1, 0])
ax1.set_facecolor(face_color)
show(inside_raster, transform=inside_transform, ax=ax1, cmap='gist_earth')
show(outside_raster, transform=outside_transform, ax=ax1, cmap='binary', norm=norm)
ax1.scatter(main_confluence[0], main_confluence[1], c='red', marker='2')
ax1.set_title('Original Elevations', color=labels_color)
ax1.set_xlabel('Longitude (°)', color=labels_color)
ax1.set_ylabel('Latitude (°)', color=labels_color)
ax1.tick_params(axis='both', colors=labels_color)

ax2 = fig.add_subplot(gs0[1, 1])
ax2.set_facecolor(face_color)
show(interpolated_inside_raster, transform=interpolated_inside_transform, ax=ax2, cmap='gist_earth')
show(outside_raster, transform=outside_transform, ax=ax2, cmap='binary', norm=norm)
ax2.scatter(main_confluence[0], main_confluence[1], c='red', marker='2')
ax2.set_title('Interpolated Elevations', color=labels_color)
ax2.set_xlabel('Longitude (°)', color=labels_color)
ax2.set_ylabel('Latitude (°)', color=labels_color)
ax2.tick_params(axis='both', colors=labels_color)

ax3 = fig.add_subplot(gs0[2, 0], projection='3d')
ax3.set_facecolor(face_color)
ax3.set_title(f"Modeling\nPlane parameters (ax + bx + c) :\na = {model[0]:.2e}, b = {model[1]:.2e}, c = {model[2]:.2e}", color=labels_color)
ax3.plot(corrected_projected_edges_points[0],corrected_projected_edges_points[1],corrected_projected_edges_points[2], c=labels_color)
c1 = ax3.scatter(corrected_projected_filtered_points[0], corrected_projected_filtered_points[1], corrected_projected_filtered_points[2], c=corrected_projected_filtered_points[2], cmap='gist_earth', s=1)
plane_colors = matplotlib.pyplot.cm.inferno((corrected_model[5] - numpy.min(corrected_model[5])) / (numpy.max(corrected_model[5]) - numpy.min(corrected_model[5])))
ax3.plot_surface(corrected_model[3], corrected_model[4], corrected_model[5], facecolors=plane_colors, alpha=0.1)
ax3.set_xlabel('Longitude (m)', color=labels_color)
ax3.set_ylabel('Latitude (m)', color=labels_color)
ax3.set_zlabel('Altitude (m)', color=labels_color)
ax3.tick_params(axis='x', colors=labels_color)
ax3.tick_params(axis='y', colors=labels_color)
ax3.tick_params(axis='z', colors=labels_color)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
# # ax3.view_init(elev=90, azim=-90)  # elev=90 to see from above

ax4 = fig.add_subplot(gs0[2, 1], projection='polar')
ax4.set_facecolor(face_color)
ax4.plot(edges_theta, edges_r, color=labels_color)
ax4.annotate("", xy=(point_min_theta, point_min_r), xytext=(point_max_theta, point_max_r), arrowprops=dict(arrowstyle="->", lw=3, color='red'),size=15)
ax4.set_title(f"Polar Projection\nSlope : {slope} m/km\nAzimuth : {round(azimuth,2)} °", color=labels_color)
ax4.set_xticklabels(['E\n90°', 'N-E\n45°', 'N\n0°', 'N-W\n315°', 'W\n270°', 'S-W\n225°', 'S\n180°', 'S-E\n135°'], color=labels_color)
for label in ax4.yaxis.get_ticklabels():
    label.set_color(labels_color)

cbar = matplotlib.pyplot.colorbar(c1, ax=[ax1, ax2, ax3, ax4])
cbar.ax.tick_params(colors=labels_color)
cbar.set_label('Altitude (m)', color=labels_color)

matplotlib.pyplot.show()
#%%



fig.savefig('./Modeling.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
