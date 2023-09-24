# ~ | MODULES AND LIBRARIES | ~ #


import h5py
import numpy
import pandas
import geopandas
import rasterio
import warnings
import glob
import os
import math
import icepyx

from math import cos, sin
from typing import List, Union, Tuple
from scipy.interpolate import griddata
from rasterio.transform import from_origin
from numba import jit
from shapely import Polygon, MultiPolygon, LineString
from rasterio.mask import mask
from sklearn.linear_model import LinearRegression
from math import atan2, radians, degrees


# ~ | SUBSCRIPTS | ~ #


def download_icesat_data(spatial_bounds: List[float],
                         time_period: Tuple[str, str],
                         destination: str = './icesatdata') -> None:
    """
    Download ICESAT data for a given spatial and temporal range.

    Parameters:
    - spatial_bounds (List[float]): The spatial extent for the data query.
    - time_period (Tuple[str, str]): The date range for the data query, in the format ['YYYY-MM-DD', 'YYYY-MM-DD'].
    - destination (str, optional): Directory path where the downloaded data will be saved.
                                   Default is './icesatdata'.

    Returns:
    None. The function will download the data to the specified destination.

    Note:
    Ensure that required credentials for Earthdata are already set up.
    """

    # Define region bounds
    x1, y1, x2, y2 = spatial_bounds
    region_bounds = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]

    # Initialize the data query for the ATL08 product
    query_region = icepyx.Query('ATL08', region_bounds, time_period)

    # Authenticate and establish a connection to Earthdata
    query_region.earthdata_login()

    # Set additional parameters for data retrieval (if any)
    query_region.subsetparams()

    # Download the data
    query_region.download_granules(destination)


def extract_geospatial_data(hdf5_file_path: str) -> List[numpy.ndarray]:
    """
    Extract geospatial data (longitudes, latitudes, and altitudes) from a given ICESat hdf5 file.

    Parameters:
    - hdf5_file_path (str): The path to the ICESat hdf5 data file.

    Returns:
    - list: A list of numpy arrays containing longitudes, latitudes, and altitudes in the respective order.

    Note:
    Assumes a specific structure and key set within the hdf5 file.
    """

    # Open hdf5 file
    hdf5_file = h5py.File(hdf5_file_path, "r")

    # Specific keywords
    typical_keys = ['METADATA', 'ancillary_data', 'ds_geosegments', 'ds_metrics', 'ds_surf_type', 'gt1l', 'gt1r',
                    'gt2l', 'gt2r', 'gt3l', 'gt3r', 'orbit_info', 'quality_assessment']

    longitudes, latitudes, altitudes = [], [], []

    # If keyword is the one we want then add data to longitudes, latitudes, altitudes
    if set(hdf5_file.keys()) == set(typical_keys):

        for key in ["gt1l", "gt2l", "gt3l", "gt1r", "gt2r", "gt3r"]:
            longitudes.extend(hdf5_file[key]["land_segments"]["longitude"][:].tolist())
            latitudes.extend(hdf5_file[key]["land_segments"]["latitude"][:].tolist())
            altitudes.extend(hdf5_file[key]["land_segments"]["dem_h"][:].tolist())

    hdf5_file.close()

    return [numpy.array(data) for data in [longitudes, latitudes, altitudes]]


def extract_data_from_folder(folder_path: str) -> List[numpy.ndarray]:
    """
    Extracts geospatial data from all ICESat hdf5 files present in the specified folder.

    Parameters:
    - folder_path (str): The path to the folder containing the ICESat hdf5 data files.

    Returns:
    - list: A list of numpy arrays containing aggregated longitudes, latitudes, and altitudes
            from all files in the folder in the respective order.

    Note:
    This function uses the read_icesatdata_file() function to process each individual file.
    Ensure this helper function is properly defined in the same environment.
    """

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Collect all .h5 files in the given folder
    hdf5_files = glob.glob(f"{folder_path}/*.h5")

    longitudes, latitudes, altitudes = [], [], []

    # Process each file and aggregate the data
    for file in hdf5_files:
        x_temp, y_temp, z_temp = extract_geospatial_data(file)

        valid_indices = ~numpy.isnan(x_temp)

        longitudes.extend(x_temp[valid_indices].tolist())
        latitudes.extend(y_temp[valid_indices].tolist())
        altitudes.extend(z_temp[valid_indices].tolist())

    return [numpy.array(data) for data in [longitudes, latitudes, altitudes]]


def rasterize_geospatial_data(data: List[numpy.ndarray],
                              output_path: str,
                              pixel_size: float) -> None:
    """
    Rasterize geospatial data and save it as a .tif file.

    Parameters:
    - data (List[numpy.ndarray]): Input geospatial data in the form [longitudes, latitudes, altitudes].
    - output_path (str): The path to for outputs.
    - pixel_size (float): The pixel size of the raster.

    Returns:
    - None: Saves the rasterized data to a .tif file.
    """

    # Extract data
    longitudes, latitudes, altitudes = data

    # Raster parameters
    min_lon, max_lon = longitudes.min(), longitudes.max()
    min_lat, max_lat = latitudes.min(), latitudes.max()
    width = int((max_lon - min_lon) / pixel_size)
    height = int((max_lat - min_lat) / pixel_size)

    # Initialize raster and count
    raster = numpy.full((height, width), numpy.nan)
    count = numpy.zeros((height, width), dtype=int)

    # Populate raster
    for lon, lat, alt in zip(longitudes, latitudes, altitudes):
        x = int((lon - min_lon) / pixel_size)
        x = min(x, width - 1)

        y = height - 1 - int((lat - min_lat) / pixel_size)
        y = min(y, height - 1)

        if numpy.isnan(raster[y, x]):
            raster[y, x] = alt
        else:
            raster[y, x] += alt
        count[y, x] += 1

    # Compute average altitude for each raster pixel
    valid_pixels = count > 0
    raster[valid_pixels] /= count[valid_pixels]

    interpolated_raster = interpolate_missing_data(raster)

    # Save raster with rasterio
    transform = from_origin(min_lon, max_lat, pixel_size, pixel_size)
    with rasterio.open(output_path + '/original_raster.tif', 'w', driver='GTiff', height=height, width=width, count=1, dtype=str(raster.dtype), crs='EPSG:4326', transform=transform, nodata=numpy.nan) as out_file:
        out_file.write(raster, 1)

    with rasterio.open(output_path + '/interpolated_raster.tif', 'w', driver='GTiff', height=height, width=width, count=1,dtype=str(interpolated_raster.dtype), crs='EPSG:4326', transform=transform, nodata=numpy.nan) as out_file:
        out_file.write(interpolated_raster, 1)


@jit(nopython=True)
def clip_geospatial_data(geospatial_data: List[numpy.ndarray],
                         bounding_edges: Tuple[float, float, float, float]) -> List[numpy.ndarray]:
    """
    Clip geospatial data based on specified bounding edges.

    Parameters:
    - geospatial_data (List[numpy.ndarray]): A list containing longitudes, latitudes, and altitudes as numpy arrays.
    - bounding_edges (Tuple[float, float, float, float]): A tuple defining left, right, bottom, and top edges for clipping.

    Returns:
    - list: A list of numpy arrays containing clipped longitudes, latitudes, and altitudes.
    """

    longitudes, latitudes, altitudes = geospatial_data

    left_edge, right_edge, bottom_edge, top_edge = bounding_edges

    longitude_mask = (longitudes >= left_edge) & (longitudes <= right_edge)
    latitude_mask = (latitudes >= bottom_edge) & (latitudes <= top_edge)
    geospatial_mask = longitude_mask & latitude_mask

    return [data[geospatial_mask] for data in [longitudes, latitudes, altitudes]]


def interpolate_missing_data(raster: numpy.ndarray,
                             method: str = 'linear') -> numpy.ndarray:
    """
    Interpolate and fill missing (NaN) values in a raster.

    Parameters:
    - raster (numpy.ndarray): Input 2D array representing the raster.
    - method (str, optional): Interpolation method. Default is 'linear'.
                              Options include 'linear', 'nearest', and 'cubic'.

    Returns:
    - numpy.ndarray: The raster with NaN values interpolated and filled.
    """

    # Create coordinate matrices for the raster
    x = numpy.arange(raster.shape[1])
    y = numpy.arange(raster.shape[0])
    X, Y = numpy.meshgrid(x, y)

    # Identify known (non-NaN) points
    known_points = numpy.array([X[~numpy.isnan(raster)], Y[~numpy.isnan(raster)]]).T
    known_values = raster[~numpy.isnan(raster)]

    # Identify points with NaN values that need interpolation
    nan_points = numpy.array([X[numpy.isnan(raster)], Y[numpy.isnan(raster)]]).T

    # Interpolate using griddata
    interpolated_values = griddata(known_points, known_values, nan_points, method=method)

    # Replace NaN values in the raster with interpolated values
    interpolated_raster = raster.copy()
    interpolated_raster[numpy.isnan(interpolated_raster)] = interpolated_values

    return interpolated_raster


def clip_geospatial_data_with_mask(geospatial_data: Union[List[numpy.ndarray], rasterio.io.DatasetReader],
                                   mask_gdf: geopandas.GeoDataFrame,
                                   buffer_distance: float,
                                   clip_type: str = 'within',
                                   region_scaling: float = 0.6) -> Union[List[numpy.ndarray], Tuple[numpy.ndarray, rasterio.transform.Affine], int]:
    """
    Clip geospatial data using a provided mask in the form of a geodataframe.
    Handles both vector (list or GeoDataFrame) and raster (rasterio DatasetReader) data.

    Parameters:
    - geospatial_data: Input geospatial data. Can be a list of arrays or a rasterio DatasetReader.
    - mask_gdf: The shapefile (as a GeoDataFrame) used for clipping.
    - buffer_distance: Distance to buffer (or shrink if negative) the clipping shape.
    - clip_type: Determines the type of clipping. 'within' for points inside the shape, and other values for points outside.
                 Default is 'within'.
    - region_scaling: Scaling factor for creating region masks for raster data. Default is 0.6.

    Returns:
    - For vector data: A list of numpy arrays containing clipped longitudes, latitudes, and altitudes.
    - For raster data: A tuple containing the clipped data and its transform.
    - Returns -1 for unsupported data types.
    """

    # Handling Vector Data
    if isinstance(geospatial_data, list):

        longitudes, latitudes, altitudes = geospatial_data
        df = pandas.DataFrame({
            'Longitude': longitudes,
            'Latitude': latitudes,
            'Altitude': altitudes
        })

        geospatial_gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
        geospatial_gdf.set_crs('EPSG:4326', inplace=True)

        buffered_mask_gdf = mask_gdf.copy()
        if clip_type == 'within':
            buffer_distance = -buffer_distance
        buffered_mask_gdf.geometry = mask_gdf.geometry.buffer(buffer_distance)

        predicate = 'within' if clip_type == 'within' else 'disjoint'
        clipped_points_gdf = geopandas.sjoin(geospatial_gdf, buffered_mask_gdf, predicate=predicate)

        clipped_longitudes = numpy.array(clipped_points_gdf.Longitude)
        clipped_latitudes = numpy.array(clipped_points_gdf.Latitude)
        clipped_altitudes = numpy.array(clipped_points_gdf.Altitude)

        return [clipped_longitudes, clipped_latitudes, clipped_altitudes]

    # Handling Raster Data
    elif isinstance(geospatial_data, rasterio.io.DatasetReader):

        left, bottom, right, top = mask_gdf.bounds.iloc[0]
        lon_diff = abs(left - right)
        lat_diff = abs(bottom - top)
        max_diff = max(lon_diff, lat_diff)
        lon_centroid = mask_gdf.centroid.x.iloc[0]
        lat_centroid = mask_gdf.centroid.y.iloc[0]

        region_mask = Polygon([(lon_centroid - region_scaling * max_diff, lat_centroid - region_scaling * max_diff),
                               (lon_centroid + region_scaling * max_diff, lat_centroid - region_scaling * max_diff),
                               (lon_centroid + region_scaling * max_diff, lat_centroid + region_scaling * max_diff),
                               (lon_centroid - region_scaling * max_diff, lat_centroid + region_scaling * max_diff)])

        buffer_mask_gdf = mask_gdf.geometry.iloc[0].buffer(buffer_distance if clip_type != 'within' else -buffer_distance)
        dataset, transform = mask(geospatial_data, [buffer_mask_gdf], invert=(clip_type != 'within'), crop=(clip_type == 'within'), nodata=numpy.nan)

        if clip_type != 'within':
            with rasterio.open('./temp.tif', 'w', driver='GTiff', height=dataset.shape[1], width=dataset.shape[2],
                               count=1, dtype=dataset.dtype, crs='EPSG:4326', transform=transform) as dst:
                dst.write(dataset[0], 1)

            with rasterio.open('./temp.tif', 'r') as temp_file:
                dataset, transform = mask(temp_file, [region_mask], crop=True)

        return dataset[0], transform

    # Unsupported Data Type
    else:
        return -1


def filter_points_in_areas(points: numpy.ndarray,
                           areas: List[numpy.ndarray]) -> List[numpy.ndarray]:
    """
    Filter and select points that lie within given areas.

    Parameters:
    - points (numpy.ndarray): Input geospatial points in the form [x_coords, y_coords].
    - areas (List[numpy.ndarray]): List of areas, where each area is defined as [x_min, x_max, y_min, y_max].

    Returns:
    - List[numpy.ndarray]: Filtered geospatial points.
    """

    x, y = points[0], points[1]

    masks = []
    for area in areas:
        x_min, x_max, y_min, y_max = area
        mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        masks.append(mask)

    combined_mask = numpy.logical_or.reduce(masks)

    return [x[combined_mask], y[combined_mask]]


def extract_confluence_points(lake_geodataframe: geopandas.GeoDataFrame,
                              path_to_rivers_shapefile: str) -> List[numpy.ndarray]:
    """
    Extract the confluence points where a given lake intersects with rivers.

    Parameters:
    - lake_geodataframe (geopandas.GeoDataFrame): GeoDataFrame representing the lake.
    - path_to_rivers_shapefile (str): Path to the shapefile containing rivers.

    Returns:
    - List[numpy.ndarray]: List containing x and y coordinates of confluence points.
    """

    # Extract the lake's edge as a LineString
    edge = geopandas.GeoDataFrame({'geometry': [LineString(list(lake_geodataframe.iloc[0].geometry.exterior.coords))]},
                                  crs='EPSG:4326')

    # Read the shapefile containing rivers
    rivers = geopandas.read_file(path_to_rivers_shapefile)

    # Filter the rivers that intersect with the lake
    rivers = rivers[rivers.intersects(lake_geodataframe.iloc[0].geometry)]

    # Convert rivers' CRS to WGS 84
    rivers = rivers.to_crs('EPSG:4326')

    # Find the intersection points between the lake's edge and the rivers
    intersection_points = edge.geometry.iloc[0].intersection(rivers.unary_union)

    # Extract x and y coordinates of the intersection points
    points_x = [point.x for point in intersection_points.geoms]
    points_y = [point.y for point in intersection_points.geoms]

    return [numpy.array(data) for data in [points_x, points_y]]


def unrasterize(raster_data: numpy.ndarray,
                geo_transform: tuple) -> numpy.ndarray:
    """
    Convert a raster dataset into lists of geospatial coordinates and their respective values.

    :param raster_data: A 2D numpy array representing the raster values. Each value corresponds to a pixel.
    :param geo_transform: A tuple of six numbers describing the affine transformation for the raster dataset.
                          The tuple structure is:
                          (pixel width, rotation (usually 0), top-left x-coordinate,
                           rotation (usually 0), pixel height (usually negative), top-left y-coordinate)

    :returns: Lists of x-coordinates, y-coordinates, and corresponding pixel values for non-NaN pixels.
    """

    longitudes, latitudes, pixel_values = [], [], []

    for row in range(raster_data.shape[0]):
        for col in range(raster_data.shape[1]):
            pixel_value = raster_data[row, col]

            # Check if the pixel value is not NaN
            if not numpy.isnan(pixel_value):
                # Compute geospatial coordinates based on the transformation
                lon = geo_transform[2] + (col + 0.5) * geo_transform[0]
                lat = geo_transform[5] + (row + 0.5) * geo_transform[4]

                # Append coordinates and pixel value to the respective lists
                longitudes.append(lon)
                latitudes.append(lat)
                pixel_values.append(pixel_value)

    return [numpy.array(data_list) for data_list in [longitudes, latitudes, pixel_values]]


def model_altitude_from_coordinates(geospatial_data: List[numpy.ndarray]) -> List[Union[float, numpy.ndarray]]:
    """
    Model the relationship between geospatial coordinates (longitudes and latitudes) and altitudes using linear regression.

    Parameters:
    - geospatial_data (List[numpy.ndarray]): A list containing arrays of longitudes, latitudes, and altitudes.

    Returns:
    - list: Model parameters and prediction grid. Contains:
        - longitude_coeff, latitude_coeff: Coefficients for longitude and latitude, respectively.
        - intercept: Model intercept.
        - xx, yy: Meshgrid of longitudes and latitudes.
        - zz: Predicted altitudes on the meshgrid.
        - r_squared: R squared value for the model.
    """

    longitudes, latitudes, altitudes = geospatial_data

    # Prepare input data for modeling
    locations = numpy.column_stack((longitudes, latitudes))

    # Initialize and fit the linear regression model
    regression_model = LinearRegression()
    regression_model.fit(locations, altitudes)

    # Extract model parameters
    longitude_coeff, latitude_coeff = regression_model.coef_
    intercept = regression_model.intercept_
    r_squared = regression_model.score(locations, altitudes)

    # Generate a meshgrid to predict altitudes over a grid
    xx, yy = numpy.meshgrid(
        numpy.linspace(longitudes.min(), longitudes.max(), 10),
        numpy.linspace(latitudes.min(), latitudes.max(), 10)
    )

    # Predict altitudes on the meshgrid
    zz = regression_model.predict(numpy.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

    return [longitude_coeff, latitude_coeff, intercept, xx, yy, zz, r_squared]


def filter_extreme_altitudes(geospatial_data: List[numpy.ndarray],
                             filter_threshold: float) -> List[numpy.ndarray]:
    """
    Filter extreme altitude values from geospatial data based on a specified filter threshold.

    Parameters:
    - geospatial_data (List[numpy.ndarray]): A list containing arrays of longitudes, latitudes, and altitudes.
    - filter_threshold (float): Threshold percentage for filtering extreme altitudes.

    Returns:
    - list: A list of numpy arrays containing filtered longitudes, latitudes, and altitudes.
    """

    longitudes, latitudes, altitudes = map(numpy.array, geospatial_data)  # Make sure they are numpy arrays

    model_params = model_altitude_from_coordinates([longitudes, latitudes, altitudes])
    longitude_coeff, latitude_coeff, intercept = model_params[:3]

    # Remove modeled trend from altitudes
    residuals = altitudes - (longitude_coeff * longitudes + latitude_coeff * latitudes + intercept)

    # Determine altitude thresholds based on the filter value
    lower_threshold, upper_threshold = numpy.percentile(residuals, [filter_threshold, 100 - filter_threshold])
    valid_altitudes_mask = (residuals >= lower_threshold) & (residuals <= upper_threshold)

    # Apply mask to filter data
    longitudes, latitudes, residuals = longitudes[valid_altitudes_mask], latitudes[valid_altitudes_mask], residuals[valid_altitudes_mask]

    # Add back the modeled trend to the residuals
    corrected_altitudes = residuals + (longitude_coeff * longitudes + latitude_coeff * latitudes + intercept)

    return [longitudes, latitudes, corrected_altitudes]


def extract_geospatial_edges(geometry: Union[Polygon, MultiPolygon]) -> List[numpy.ndarray]:
    """
    Extract the exterior coordinates (edges) of the provided geometry.

    Parameters:
    - geometry (Union[Polygon, MultiPolygon]): The input geometry (either a Polygon or a MultiPolygon).

    Returns:
    - list: A list containing arrays of longitudes and latitudes representing the edges.
    """

    # Extract exterior coordinates based on geometry type
    if geometry.type == 'Polygon':
        contour_points = list(geometry.exterior.coords)
    elif geometry.type == 'MultiPolygon':
        # Combine exterior coords of all individual polygons in the MultiPolygon
        contour_points = [coord for polygon in geometry for coord in list(polygon.exterior.coords)]
    else:
        raise ValueError("Unsupported geometry type. Only 'Polygon' and 'MultiPolygon' are accepted.")

    longitudes, latitudes = zip(*contour_points)

    return [numpy.array(longitudes), numpy.array(latitudes)]


def transform_coordinates(points_data,
                          crs='EPSG:7853'):
    """
    Transform the CRS of a given point or set of points.

    :param points_data: List or tuple of coordinates.
                        Single point format: (lon, lat)
                        Multiple points format: ([lons], [lats])
    :param crs: The target coordinate reference system. Default is 'EPSG:7853'.

    :returns: Transformed coordinates. Single point: (x, y). Multiple points: ([xs], [ys])
    """

    single_point = False
    if isinstance(points_data[0], (int, float)) and isinstance(points_data[1], (int, float)):
        single_point = True
        points_data = [[points_data[0]], [points_data[1]]]

    df = pandas.DataFrame({'Longitude': points_data[0], 'Latitude': points_data[1]})
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

    gdf = gdf.set_crs('EPSG:4326').to_crs(crs)

    x_new, y_new = gdf.geometry.x, gdf.geometry.y

    if single_point:
        return x_new.iloc[0], y_new.iloc[0]
    return [numpy.array(data) for data in [x_new, y_new]]


def generate_lake_information(lake_gdf: geopandas.GeoDataFrame,
                              time_period: List[str],
                              resolution: float,
                              filter_threshold: int,
                              target_crs: str,
                              r_squared_value: float,
                              slope: float,
                              azimuth: float) -> pandas.DataFrame:
    """
    Generate a DataFrame containing lake's geospatial and metadata.

    :param lake_gdf: GeoDataFrame containing the lake's geographical data.
    :param time_period: A list of two strings indicating the start and end dates.
    :param resolution: Spatial resolution of the data in degrees per pixel.
    :param filter_threshold: Percentage threshold used in data filtering.
    :param target_crs: Coordinate reference system for the area and perimeter calculations.
    :param r_squared_value: Coefficient of determination for some model (e.g., regression).
    :param slope: Slope value in meters per kilometer.
    :param azimuth: Azimuth of the slope in degrees.

    :returns: DataFrame with the lake's information.
    """

    # Compute the centroid and format as a string
    centroid_coords = [lake_gdf.centroid.x.iloc[0], -lake_gdf.centroid.y.iloc[0]]
    location = f"{centroid_coords[0]:.3f} ° E - {centroid_coords[1]:.3f} ° S"

    # Convert to the target CRS and calculate area and perimeter
    transformed_gdf = lake_gdf.to_crs(target_crs)
    area_km2 = round(transformed_gdf.area.iloc[0] / 1_000_000)
    perimeter_km = round(transformed_gdf.length.iloc[0] / 1_000)

    # Organize data for DataFrame creation
    data = {
        "Initial parameters": ["Lake Name", "Location", "Date Range", "Resolution", "Filter Threshold"],
        "Values1": [lake_gdf.Lake_Name.iloc[0], location, f"{time_period[0]} to {time_period[1]}",
                    f"{resolution} °/pixel", f"{filter_threshold} %"],
        "Results": ["Area (km²)", "Perimeter (km)", "R² (%)", "Gradient (m/km)", "Azimuth (°)"],
        "Values2": [area_km2, perimeter_km, f"{round(r_squared_value)} %", f"{slope} m/km", f"{azimuth} °"]
    }

    return pandas.DataFrame(data)


def find_intersection_points(linestring_geodataframe: geopandas.GeoDataFrame,
                             mask_geodataframe: geopandas.GeoDataFrame) -> Tuple[List[float], List[float]]:
    """
    Identify the intersection points between a GeoDataFrame of LineStrings and the edge of a mask (region or area).

    Parameters:
    - linestring_geodataframe (geopandas.GeoDataFrame): The input GeoDataFrame with LineStrings.
    - mask_geodataframe (geopandas.GeoDataFrame): The mask or region GeoDataFrame.

    Returns:
    - Tuple[List[float], List[float]]: Lists of x and y coordinates of intersection points.
    """

    # Extract the exterior boundary of the mask as a LineString
    edge = geopandas.GeoSeries(LineString(list(mask_geodataframe.iloc[0].geometry.exterior.coords)))

    # Find the intersection points between the linestrings and the mask's edge
    intersection_points = linestring_geodataframe.geometry.intersection(edge.unary_union)

    x, y = [], []
    for geometry in intersection_points:
        if geometry.geom_type == 'Point':
            x.append(geometry.x)
            y.append(geometry.y)
        elif geometry.geom_type == 'MultiPoint':
            for point in geometry:
                x.append(point.x)
                y.append(point.y)

    return numpy.array(x), numpy.array(y)


def calculate_initial_compass_bearing(pointA: tuple,
                                      pointB: tuple) -> float:
    """
    Calculate the azimuth between two geospatial points.

    The result falls into the range [0, 360).

    :param pointA: Tuple (longitude, latitude) of the starting point.
    :param pointB: Tuple (longitude, latitude) of the destination point.

    :returns: Azimuth in degrees.
    """

    lat1 = radians(pointA[1])
    lat2 = radians(pointB[1])

    diffLong = radians(pointB[0] - pointA[0])

    x = atan2(sin(diffLong) * cos(lat2),
              cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(diffLong))

    x = degrees(x)
    compass_bearing = (x + 360) % 360

    return compass_bearing


def convert_to_polar_coordinates(data: numpy.ndarray,
                                 reference_point: numpy.ndarray) -> numpy.ndarray:
    """
    Convert Cartesian coordinates to Polar coordinates.

    Parameters:
    - data (numpy.ndarray): Input data in the form [x_coords, y_coords].
    - reference_point (numpy.ndarray): Reference point [x, y], usually the origin for the Polar system.

    Returns:
    - numpy.ndarray: Polar coordinates [r, theta].
    """

    data_x, data_y = data
    reference_x, reference_y = reference_point

    # Calculate the radial distance for each data point
    r = numpy.sqrt((data_x - reference_x) ** 2 + (data_y - reference_y) ** 2)

    # Calculate the angle (in radians) for each data point
    theta = numpy.arctan2(data_y - reference_y, data_x - reference_x)

    return r, theta


def convert_hgt_to_tiff(folder_path: str, left_longitude: float, top_latitude: float) -> int:
    """
    Convert .hgt files in a folder to a GeoTIFF format.

    :param folder_path: Path to the folder containing .hgt files.
    :param left_longitude: The leftmost longitude of the bounding box.
    :param top_latitude: The topmost latitude of the bounding box.

    :returns: 0 if successful, -1 otherwise.
    """

    hgt_files = glob.glob(os.path.join(folder_path, '*.hgt'))

    # Validate the number of files
    if len(hgt_files) not in [4, 9]:
        return -1

    combined_data = []

    for file_path in hgt_files:
        file_size = os.path.getsize(file_path)
        dimension = int(math.sqrt(file_size / 2))

        assert dimension * dimension * 2 == file_size, 'Invalid file size'
        data = numpy.fromfile(file_path, numpy.dtype('>i2'), dimension * dimension).reshape((dimension, dimension))
        combined_data.append(data)

    # Combine the data based on the number of files
    if len(hgt_files) == 4:
        combined_data = numpy.block([[combined_data[0], combined_data[1]],
                                     [combined_data[2], combined_data[3]]])
    elif len(hgt_files) == 9:
        combined_data = numpy.block([[combined_data[0], combined_data[1], combined_data[2]],
                                     [combined_data[3], combined_data[4], combined_data[5]],
                                     [combined_data[6], combined_data[7], combined_data[8]]])

    resolution = len(hgt_files) / combined_data.shape[0]
    output_file = os.path.join(folder_path, 'shuttle.tif')

    with rasterio.open(output_file, 'w', driver='GTiff', height=combined_data.shape[0], width=combined_data.shape[1],
                       count=1, dtype=numpy.float64, crs='EPSG:4326',
                       transform=rasterio.transform.from_origin(left_longitude, top_latitude, resolution,
                                                                resolution)) as dst:
        dst.write(combined_data, 1)

    return 0
