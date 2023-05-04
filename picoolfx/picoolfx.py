# Nicola Cerutti, 2023
# See LICENSE for details

"""picoolfx - A library for applying cool effects to images."""

__version__ = "0.1.0"

import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import geopandas as gpd
from shapely.geometry import LineString
from .utils import (
    buffered_intersections,
    coords_to_gdf_spiral,
    create_noise_matrix,
    flow_polygons,
    polygony,
    spiral_coords,
)


def check_image(img: Image.Image) -> bool:
    """Checks if the image is a square

    Args:
        img (Image): PIL image

    Returns:
        bool: True if the image is a square, False otherwise
    """
    width, height = img.size
    if width != height:
        return False
    return True


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
    image = image.resize((x_pixels, y_pixels))
    image = image.resize((width, height), Image.NEAREST)
    return image


def spiralise(
    input_image=None,
    spiral_points=5000,
    spiral_turns=50,
    spiral_r0=0,
    spiral_r1_f=0.5,
    thin=0.00025,
    thick_f=0.95,
    spiral_offset_angle=0,
    color="#000000",
    colormap="gray",
    rescaler_factor=1.0,
    alpha=0.75,
):
    """
    Args:
        image (PIL Image): Image to spiralise
        spiral_points (int, optional): Number of points in the spiral. Defaults to 5000.
        spiral_turns (int, optional): Number of turns in the spiral. Defaults to 50.
        spiral_r0 (int, optional): Initial radius of the spiral. Defaults to 0.
        spiral_r1_f (int, optional): Final radius of the spiral. Defaults to 0.5.
        thin (float, optional): Minimum thickness of the spiral. Defaults to 0.00025.
        thick_f (float, optional): Maximum thickness of the spiral. Defaults to 0.95.
        spiral_offset_angle (int, optional): Angle to offset the spiral, in degrees. Defaults to 0.
        color (str, optional): Color of the spiral. Defaults to "#000000".
        colormap (str, optional): Colormap of the image. Defaults to "gray".
        rescaler_factor (float, optional): Rescaler factor for the image. Defaults to 1.0.
        alpha (float, optional): Alpha value of the image. Defaults to 0.75.
    """
    if input_image is None:
        raise ValueError("No image provided.")
    if not check_image(input_image):
        raise ValueError(
            "Invalid image format. The spiralise function only accepts square PIL images."
        )
    polygons_gdf = polygony(input_image, rescaler_factor=rescaler_factor)
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
    if intersections is not None:
        if colormap == "none":
            intersections.plot(ax=ax, facecolor=color, edgecolor="none", alpha=alpha)
        else:
            intersections.plot(
                ax=ax, facecolor=color, edgecolor="none", cmap=colormap, alpha=alpha
            )
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    spiral_image = Image.open(buf)
    plt.close(fig)  # Close the figure to prevent it from being displayed
    return spiral_image


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


# def double_spiral_function(
#     input_image_1="test_a.png",
#     input_image_2="test_b.png",
#     size=300,
#     n_shades=16,
#     spiral_points=5000,
#     spiral_turns=50,
#     spiral_r0=0,
#     spiral_r1_f=0.5,
#     thin=0.00025,
#     thick_f=0.5,
#     spiral_offset_angle=0,
#     crop=False,
#     color_1="gray",
#     color_2="gray",
#     alpha_1=0.75,
#     alpha_2=0.5,
#     output_image="output.png",
#     rescaler_factor=1.0,
# ):
#     # Prepare the image
#     img_a = Image.open(input_image_1)
#     img_a = prepare_image(img_a, size=size, shades=n_shades, crop=crop)
#     polygons_gdf_a = polygony(img_a, rescaler_factor=rescaler_factor)
#     img_b = Image.open(input_image_2)
#     img_b = prepare_image(img_b, size=size, shades=n_shades, crop=crop)
#     polygons_gdf_b = polygony(img_b, rescaler_factor=rescaler_factor)
#     try:
#         bounds = polygons_gdf_a.total_bounds
#     except ValueError:
#         print("ValueError: No polygons found.")
#         return None
#     width = bounds[2] - bounds[0]
#     height = bounds[3] - bounds[1]
#     center_x = (bounds[0] + bounds[2]) / 2
#     center_y = (bounds[1] + bounds[3]) / 2
#     scale_factor = max(width, height)
#     coords = spiral_coords(
#         center_x,
#         center_y,
#         spiral_points,
#         spiral_turns,
#         spiral_r0,
#         spiral_r1_f,
#         spiral_offset_angle,
#         scale=scale_factor,
#     )
#     gdf_spiral = coords_to_gdf_spiral(coords)
#     intersections_positive = buffered_intersections(
#         polygons_gdf_a,
#         gdf_spiral,
#         spiral_turns,
#         scale_factor,
#         thin,
#         thick_f,
#         spiral_r1=spiral_r1_f,
#     )

#     # Create intersections with positive and negative buffer values
#     intersections_positive["n"] = intersections_positive["col"].apply(
#         lambda x: (thick_f - thin) * x + thin
#     )

#     intersections_positive["geometry"] = intersections_positive.geometry.buffer(
#         intersections_positive["n"], cap_style=2, single_sided=True
#     )

#     intersections_negative = buffered_intersections(
#         polygons_gdf_b,
#         gdf_spiral,
#         spiral_turns,
#         scale_factor,
#         thin,
#         thick_f,
#         spiral_r1=spiral_r1_f,
#     )

#     # intersections_negative["geometry"] = intersections_negative.geometry.buffer(
#     #     -intersections_negative["n"], cap_style=2, single_sided=True
#     # )

#     # Remove Points from the intersections_negative
#     intersections_negative = intersections_negative[
#         intersections_negative["geometry"].apply(lambda x: not isinstance(x, Point))
#     ]

#     intersections_positive = intersections_positive[intersections_positive.is_valid]
#     intersections_negative = intersections_negative[intersections_negative.is_valid]

#     # Plot intersections with different colors
#     fig, ax = plt.subplots()
#     intersections_positive.plot(
#         ax=ax, facecolor=color_1, edgecolor="none", alpha=alpha_1
#     )
#     intersections_negative.plot(
#         ax=ax, facecolor=color_2, edgecolor="none", alpha=alpha_2
#     )
#     ax.set_aspect("equal")
#     ax.set_axis_off()
#     plt.tight_layout()
#     plt.show()
#     fig.savefig(output_image, dpi=300, bbox_inches="tight", pad_inches=0)


def flowalise(
    input_image: Image.Image,
    x_side=300,
    y_side=300,
    n_points=800,
    step_length=1,
    n_steps=400,
    thin=0.0001,
    thick=0.25,
    rescaler_factor=1.0,
    color="black",
    alpha=1.0,
    colormap="none",
) -> Image.Image:
    """
    Creates a flow map effect on the input image using a (mostly) custom algorithm.

    Args:
        input_image (Optional[Image.Image], default=None): The input image to apply the effect to.
        x_side (int, default=300): Width of the noise matrix.
        y_side (int, default=300): Height of the noise matrix.
        n_points (int, default=800): Number of starting points for the flow lines.
        step_length (int, default=1): Length of each step in the flow lines.
        n_steps (int, default=400): Number of steps to take for each flow line.
        thin (float, default=0.0001): Minimum line width for the flow lines.
        thick (float, default=0.25): Maximum line width for the flow lines.
        rescaler_factor (float, default=1.0): Rescaling factor for the input image.
        color (str, default="black"): Color of the flow lines.
        alpha (float, default=1.0): Transparency of the flow lines.
        colormap (str, default="none"): Colormap to use for coloring the flow lines.

    Returns:
        Image.Image: The input image with the flow map effect applied.

    Raises:
        ValueError: If input_image is None.
    """
    if input_image is None:
        raise ValueError("No input image provided.")
    polygons_gdf = polygony(input_image, rescaler_factor=rescaler_factor)
    noise_matrix = create_noise_matrix(x_side, y_side, seed=np.random.randint(0, 1000))

    x_starts = np.random.uniform(1, noise_matrix.shape[1], n_points)
    y_starts = np.random.uniform(1, noise_matrix.shape[0], n_points)

    flow_lines = []

    for x_start, y_start in zip(x_starts, y_starts):
        coords = flow_polygons(x_start, y_start, step_length, n_steps, noise_matrix)
        if coords is not None and len(coords) > 1:
            line = LineString(coords)
            flow_lines.append(line)

    flow_gdf = gpd.GeoDataFrame({"geometry": flow_lines})

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
