import unittest
import numpy as np
from PIL import Image, ImageChops
from picoolfx.picoolfx import pixelate, spiralise, flowalise


class TestImageFunctions(unittest.TestCase):
    """
    TestImageFunctions is a unit testing class for the pixelate, spiralise, and flowalise functions.
    """

    def test_pixelate(self):
        """
        Test the pixelate function with a simple 4x4 black and white image, and compare the output
        with an expected result.
        """
        input_image = Image.new("1", (4, 4))
        for x in range(4):
            for y in range(4):
                input_image.putpixel((x, y), (x + y) % 2)

        # Test pixelate function
        pixelated_image = pixelate(input_image, 2)

        # Create the expected 4x4 pixelated output image
        expected_output_image = Image.new("1", (4, 4))
        for x in range(4):
            for y in range(4):
                expected_output_image.putpixel((x, y), (x // 2 + y // 2) % 2)

        # Check if the pixelated_image is equal to the expected_output_image
        self.assertTrue(
            ImageChops.difference(pixelated_image, expected_output_image).getbbox()
            is None
        )

    def test_spiralise(self):
        """
        Test the spiralise function with a simple 100x100 black and white image, and verify the
        output image's size and type.
        """
        input_image = Image.new("1", (100, 100), color=1)

        # Test spiralise function
        spiralised_image = spiralise(
            input_image,
            spiral_points=1000,
            spiral_turns=5,
            spiral_r0=10,
            spiral_r1_f=0.8,
        )

        # Verify the output image
        self.assertIsNotNone(spiralised_image)
        self.assertIsInstance(spiralised_image, Image.Image)
        self.assertEqual(spiralised_image.size, (100, 100))

    def test_flowalise(self):
        """
        Test the flowalise function with a simple 100x100 black and white image, and verify the
        output image's size and type.
        """
        input_image = Image.new("1", (100, 100), color=1)

        # Test flowalise function
        flowed_image = flowalise(
            input_image, n_points=100, step_length=0.5, n_steps=100
        )

        # Verify the output image
        self.assertIsNotNone(flowed_image)
        self.assertIsInstance(flowed_image, Image.Image)
        self.assertEqual(flowed_image.size, (100, 100))


if __name__ == "__main__":
    unittest.main()
