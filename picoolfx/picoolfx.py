""""""
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio.features
import geopandas as gpd
from shapely.geometry import LineString, Point
from utils import (
    buffered_intersections,
    coords_to_gdf_spiral,
    create_noise_matrix,
    polygony,
    prepare_image,
    spiral_coords,
)


def pixelate(image, pixel_size):
    """Pixelate the given image.

    Args:
        image (PIL.Image): Image to pixelate
        pixel_size (int): Size of the pixels

    Returns:
        PIL.Image: Pixelated image
    """
    width, height = image.size
    x_pixels = width // pixel_size
    y_pixels = height // pixel_size
    image = prepare_image(image, width, 16, crop=False, invert=False)
    image = image.resize((x_pixels, y_pixels))
    image = image.resize((width, height), Image.NEAREST)
    return image


def spiral_function(
    input_image="test.png",
    size=300,
    n_shades=16,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=0.5,
    thin=0.00025,
    thick_f=0.95,
    spiral_offset_angle=0,
    crop=False,
    color="#000000",
    colormap="gray",
    output_image="output.png",
    rescaler_factor=1.0,
    alpha=0.75,
):
    """
    Args:
        image (_type_): _description_
        n_turns (int, optional): _description_. Defaults to 50.
        n_points (int, optional): _description_. Defaults to 5000.
        size (int, optional): _description_. Defaults to 300.
        invert (_type_, optional): _description_. Defaults to FALSE.
        size (int, optional): _description_. Defaults to 300.
        n_shades (int, optional): _description_. Defaults to 16.
        spiral_points (int, optional): _description_. Defaults to 5000.
        spiral_turns (int, optional): _description_. Defaults to 50.
        spiral_r0 (int, optional): _description_. Defaults to 0.
        spiral_r1_f (int, optional): _description_. Defaults to 1.
        thin (float, optional): _description_. Defaults to 0.00025.
        thick_f (float, optional): _description_. Defaults to 0.95.
        spiral_offset_angle (int, optional): _description_. Defaults to 0.
        crop (bool, optional): _description_. Defaults to False.
        colormap (str, optional): _description_. Defaults to "gray".
        output_image (str, optional): _description_. Defaults to "output.png".
        rescaler_factor (float, optional): _description_. Defaults to 1.0.
    """
    # Prepare the image
    img = Image.open(input_image)
    img = prepare_image(img, size=size, shades=n_shades, crop=crop)
    polygons_gdf = polygony(img, rescaler_factor=rescaler_factor)
    try:
        bounds = polygons_gdf.total_bounds
    except ValueError:
        print("ValueError: No polygons found.")
        return None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    scale_factor = max(width, height)
    coords = spiral_coords(
        center_x,
        center_y,
        spiral_points,
        spiral_turns,
        spiral_r0,
        spiral_r1_f,
        spiral_offset_angle,
        scale=scale_factor,
    )
    gdf_spiral = coords_to_gdf_spiral(coords)
    intersections = buffered_intersections(
        polygons_gdf,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )
    fig, ax = plt.subplots()
    if colormap == "none":
        intersections.plot(ax=ax, facecolor=color, edgecolor="none", alpha=alpha)
    else:
        intersections.plot(
            ax=ax, facecolor=color, edgecolor="none", cmap=colormap, alpha=alpha
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def double_spiral_function(
    input_image_1="test_a.png",
    input_image_2="test_b.png",
    size=300,
    n_shades=16,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=0.5,
    thin=0.00025,
    thick_f=0.5,
    spiral_offset_angle=0,
    crop=False,
    color_1="gray",
    color_2="gray",
    alpha_1=0.75,
    alpha_2=0.5,
    output_image="output.png",
    rescaler_factor=1.0,
):
    # Prepare the image
    img_a = Image.open(input_image_1)
    img_a = prepare_image(img_a, size=size, shades=n_shades, crop=crop)
    polygons_gdf_a = polygony(img_a, rescaler_factor=rescaler_factor)
    img_b = Image.open(input_image_2)
    img_b = prepare_image(img_b, size=size, shades=n_shades, crop=crop)
    polygons_gdf_b = polygony(img_b, rescaler_factor=rescaler_factor)
    try:
        bounds = polygons_gdf_a.total_bounds
    except ValueError:
        print("ValueError: No polygons found.")
        return None
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    scale_factor = max(width, height)
    coords = spiral_coords(
        center_x,
        center_y,
        spiral_points,
        spiral_turns,
        spiral_r0,
        spiral_r1_f,
        spiral_offset_angle,
        scale=scale_factor,
    )
    gdf_spiral = coords_to_gdf_spiral(coords)
    intersections_positive = buffered_intersections(
        polygons_gdf_a,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )

    # Create intersections with positive and negative buffer values
    intersections_positive["n"] = intersections_positive["col"].apply(
        lambda x: (thick_f - thin) * x + thin
    )

    intersections_positive["geometry"] = intersections_positive.geometry.buffer(
        intersections_positive["n"], cap_style=2, single_sided=True
    )

    intersections_negative = buffered_intersections(
        polygons_gdf_b,
        gdf_spiral,
        spiral_turns,
        scale_factor,
        thin,
        thick_f,
        spiral_r1=spiral_r1_f,
    )

    # intersections_negative["geometry"] = intersections_negative.geometry.buffer(
    #     -intersections_negative["n"], cap_style=2, single_sided=True
    # )

    # Remove Points from the intersections_negative
    intersections_negative = intersections_negative[
        intersections_negative["geometry"].apply(lambda x: not isinstance(x, Point))
    ]

    intersections_positive = intersections_positive[intersections_positive.is_valid]
    intersections_negative = intersections_negative[intersections_negative.is_valid]

    # Plot intersections with different colors
    fig, ax = plt.subplots()
    intersections_positive.plot(
        ax=ax, facecolor=color_1, edgecolor="none", alpha=alpha_1
    )
    intersections_negative.plot(
        ax=ax, facecolor=color_2, edgecolor="none", alpha=alpha_2
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def grid_function(
    input_image="test.png",
    size=300,
    n_shades=16,
    grid_size=10,
    thin=0.00025,
    thick_f=0.95,
    grid_angle=0,
    crop=False,
    colormap="gray",
    output_image="output.png",
    rescaler_factor=1.0,
):
    """TBD

    Args:
        input_image (str, optional): _description_. Defaults to "test.png".
        size (int, optional): _description_. Defaults to 300.
        n_shades (int, optional): _description_. Defaults to 16.
        grid_size (int, optional): _description_. Defaults to 10.
        thin (float, optional): _description_. Defaults to 0.00025.
        thick_f (float, optional): _description_. Defaults to 0.95.
        grid_angle (int, optional): _description_. Defaults to 0.
        crop (bool, optional): _description_. Defaults to False.
        colormap (str, optional): _description_. Defaults to "gray".
        output_image (str, optional): _description_. Defaults to "output.png".
        rescaler_factor (float, optional): _description_. Defaults to 1.0.
    """
    pass


def flow_function(
    input_image: Image.Image,
    size=300,
    x_side=300,
    y_side=300,
    n_points=800,
    step_length=1,
    n_steps=400,
    n_shades=16,
    thin=0.0001,
    thick=0.0025,
    output_image="output_flow.png",
    crop=False,
    rescaler_factor=1.0,
    color="black",
    alpha=1.0,
    colormap="none",
) -> Image.Image:
    # Prepare the image
    img = prepare_image(input_image, size=size, shades=n_shades, crop=crop)
    polygons_gdf = polygony(img, rescaler_factor=rescaler_factor)

    noise_matrix = create_noise_matrix(x_side, y_side)

    x_starts = np.random.uniform(1, noise_matrix.shape[1], n_points)
    y_starts = np.random.uniform(1, noise_matrix.shape[0], n_points)

    flow_lines = []

    for x_start, y_start in zip(x_starts, y_starts):
        coords = flow_polygons(x_start, y_start, step_length, n_steps, noise_matrix)
        if coords is not None and len(coords) > 1:
            line = LineString(coords)
            flow_lines.append(line)

    flow_gdf = gpd.GeoDataFrame(geometry=flow_lines)

    # Intersect flow lines with polygons
    intersections = gpd.overlay(
        polygons_gdf, flow_gdf, how="intersection", keep_geom_type=False
    )

    # Calculate line widths based on the 'col' value
    intersections["n"] = intersections["col"].apply(lambda x: (thick - thin) * x + thin)
    intersections["geometry"] = intersections.geometry.buffer(
        intersections["n"], cap_style=2
    )

    # Plot the intersections
    fig, ax = plt.subplots()
    if colormap == "none":
        intersections.plot(ax=ax, facecolor=color, edgecolor="none", alpha=alpha)
    else:
        intersections.plot(
            ax=ax, facecolor=color, edgecolor="none", cmap=colormap, alpha=alpha
        )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    # Convert the plotted figure to a PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    flowed_image = Image.open(buf)

    plt.close(fig)  # Close the figure to prevent it from being displayed

    return flowed_image
