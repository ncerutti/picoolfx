import geopandas as gpd
import noise
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import rasterio
import rasterio.features
from shapely.geometry import LineString, shape


def buffered_intersections(
    polygons_gdf, gdf_spiral, n_turns, scale_factor, thin, thick, spiral_r1
):
    """Calculate buffered intersections between polygons and spiral.

    Args:
        polygons_gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons
        gdf_spiral (geopandas.GeoDataFrame): GeoDataFrame containing the spiral
        n_turns (int): Number of turns in the spiral
        scale_factor (float): Scaling factor for the spiral
        thin (float): Minimum buffer width
        thick (float): Maximum buffer width
        spiral_r1 (float): Final radius of the spiral

    Returns:
        geopandas.GeoDataFrame: Buffered intersections"""
    intersections = gpd.overlay(
        polygons_gdf, gdf_spiral, how="intersection", keep_geom_type=False
    )

    if not intersections.empty:
        thin = thin
        thick = ((spiral_r1 / n_turns) / 2) * thick * scale_factor

        intersections["n"] = intersections["col"].apply(
            lambda x: (thick - thin) * x + thin
        )
        intersections["geometry"] = intersections.geometry.buffer(
            intersections["n"], cap_style=2
        )

        return intersections
    else:
        return None


def check_image(img: Image.Image) -> bool:
    """Checks if the image is a square

    Args:
        img (Image): PIL image

    Returns:
        bool: True if the image is a square, False otherwise
    """
    if img is None:
        return False
    width, height = img.size
    if width != height:
        return False
    return True


def coords_to_gdf_spiral(coords):
    """Converts the given coordinates into a GeoDataFrame containing a LineString.

    Args:
        xo (float): X-coordinate of the spiral's center
        yo (float): Y-coordinate of the spiral's center
        n_points (int): Number of points in the spiral
        n_turns (int): Number of turns in the spiral
        r0 (float): Initial radius of the spiral
        r1 (float): Final radius of the spiral
        offset_angle (float): Angle to offset the spiral, in degrees
        scale (float, optional): Scaling factor for the spiral. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the coordinates of the spiral
    """

    # Convert the coordinates to a LineString
    spiral_linestring = LineString(coords)

    # Create a GeoDataFrame from the LineString
    gdf_spiral = gpd.GeoDataFrame(geometry=[spiral_linestring], crs="EPSG:4326")
    return gdf_spiral


def create_noise_matrix(x_side, y_side):
    """
    Create a noise matrix using Perlin noise.

    Args:
        x_side (int): Width of the noise matrix.
        y_side (int): Height of the noise matrix.

    Returns:
        noise_matrix (numpy array): 2D noise matrix with the given dimensions.
    """
    noise_matrix = np.zeros((y_side, x_side))

    for i in range(y_side):
        for j in range(x_side):
            noise_matrix[i][j] = noise.snoise2(
                i * 0.0003, j * 0.0003, octaves=1, persistence=0.5, lacunarity=2.0
            )

    noise_matrix = np.interp(
        noise_matrix, (noise_matrix.min(), noise_matrix.max()), (-90, 90)
    )
    return noise_matrix


def flow_polygons(x_start, y_start, step_length, n_steps, angle_matrix):
    out_x = [x_start] + [np.nan] * n_steps
    out_y = [y_start] + [np.nan] * n_steps

    if (
        x_start > angle_matrix.shape[1]
        or x_start < 1
        or y_start > angle_matrix.shape[0]
        or y_start < 1
    ):
        return None

    for i in range(n_steps):
        a = angle_matrix[int(round(out_y[i])) - 1, int(round(out_x[i])) - 1]
        step_x = np.cos(np.radians(a)) * step_length
        step_y = np.sin(np.radians(a)) * step_length

        next_x = out_x[i] + step_x
        next_y = out_y[i] + step_y

        if (
            next_x > angle_matrix.shape[1]
            or next_x < 1
            or next_y > angle_matrix.shape[0]
            or next_y < 1
        ):
            break

        out_x[i + 1] = next_x
        out_y[i + 1] = next_y

    coords = np.column_stack((out_x, out_y))
    coords = coords[~np.isnan(coords).any(axis=1)]

    return coords


def plot_polygons(polygons):
    """Plot the given polygons GeoDataFrame. Mostly for debugging.

    Args:
        polygons (geopandas.GeoDataFrame): GeoDataFrame containing polygons
    """
    fig, ax = plt.subplots()
    polygons.plot(column="col", cmap="viridis_r", ax=ax, edgecolor="none")
    plt.show()


def plot_spiral_and_polygons(spiral_coords, polygons_gdf):
    """Plot the given spiral coordinates and polygons GeoDataFrame. Mostly for debugging.

    Args:
    spiral_coords (pd.DataFrame): DataFrame containing the spiral coordinates
    polygons_gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons

    """

    # Convert the spiral coordinates to a LineString
    spiral = LineString(spiral_coords.values)

    # Create a GeoDataFrame with the spiral
    gdf_spiral = gpd.GeoDataFrame({"geometry": [spiral]})

    # Plot the polygons and spiral
    fig, ax = plt.subplots()
    polygons_gdf.plot(
        ax=ax, column="col", cmap="viridis_r", alpha=0.75, edgecolor="none"
    )
    gdf_spiral.plot(ax=ax, color="black", linewidth=1)

    # Customize the plot appearance
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def polygony(image: Image, rescaler_factor=1.0) -> list:
    """Convert the input PIL Image into a GeoDataFrame of polygons.

    Args:
        image (PIL Image): Input image
    rescaler_factor (float, optional): Rescaling factor for rescaled_red_band. Defaults to 1.0.

    Returns:
        list: List of polygons as a GeoDataFrame object
    """
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save("test2.png")
    with rasterio.open("test2.png") as src:
        red_band = src.read(1)
        # rescaled_red_band = 1 - (red_band - red_band.min()) / (
        #     red_band.max() - red_band.min()
        # )
        rescaled_red_band = rescaler_factor - (red_band - red_band.min()) / (
            red_band.max() - red_band.min()
        )
    i_sf = raster_to_geodataframe(src, rescaled_red_band)
    return i_sf


def prepare_image(
    img: Image.Image, size: int, shades: int, crop=False, invert=False
) -> Image.Image:
    """_summary_

    Args:
        img (PIL image): input image
        size (int): target size
        shades (int): number of shades
        crop (bool, optional): Should the image be cropped instead of resized? Defaults to False.
        invert (bool, optional): Should the image be inverted? Defaults to False.

    Returns:
        i (PIL image): black and white, quantized, resized image
    """
    if crop:
        width, height = img.size
        if width < size or height < size:
            raise ValueError("Image is too small to crop")
        left = (width - size) // 2
        top = (height - size) // 2
        right = (width + size) // 2
        bottom = (height + size) // 2
        i = img.crop((left, top, right, bottom))
    else:
        i = img.resize((size, size))
    i = i.quantize(shades)
    i = i.convert("L")
    if invert:
        i = ImageOps.invert(i)
    return i


def raster_to_geodataframe(image, rescaled_red_band):
    """Use rasterio to convert a raster image to a GeoDataFrame

    Args:
        image (_type_): PIL image
        rescaled_red_band (_type_): one of the channels

    Returns:
        GeoDataFrame object
    """
    mask = rescaled_red_band > 0
    shapes = rasterio.features.shapes(
        rescaled_red_band.astype(np.float32), mask=mask, transform=image.transform
    )
    polygons = [{"geometry": shape(s), "col": v} for s, v in shapes]
    return gpd.GeoDataFrame(polygons)


def spiral_coords(xo, yo, n_points, n_turns, r0, r1, offset_angle, scale=1):
    """Generate the coordinates for a spiral.

    Args:
        xo (float): X-coordinate of the spiral's center
        yo (float): Y-coordinate of the spiral's center
        n_points (int): Number of points in the spiral
        n_turns (int): Number of turns in the spiral
        r0 (float): Initial radius of the spiral
        r1 (float): Final radius of the spiral
        offset_angle (float): Angle to offset the spiral, in degrees
        scale (float, optional): Scaling factor for the spiral. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing the coordinates of the spiral
    """
    b = (r1 - r0) / (2 * np.pi * n_turns)
    l = np.linspace(0, 2 * np.pi * n_turns, num=n_points)

    x = (r0 + (b * l)) * np.cos(l + np.radians(offset_angle)) * scale + xo
    y = (r0 + (b * l)) * np.sin(l + np.radians(offset_angle)) * scale + yo

    return pd.DataFrame({"x": x, "y": y})
