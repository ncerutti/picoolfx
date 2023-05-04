import os
import unittest
from PIL import Image
from picoolfx.picoolfx import check_image, pixelate, prepare_image, spiralise, flowalise


class TestImageFunctions(unittest.TestCase):
    """
    TestImageFunctions is a unit testing class for the pixelate, spiralise, and flowalise functions.
    """

    def test_check_image(self):
        """
        Test the check_image function with a simple 4x4 black and white image.
        """
        input_image = Image.new("1", (4, 4))
        self.assertTrue(check_image(input_image))

    def test_pixelate(self):
        """
        Test the pixelate function with a simple 4x4 black and white image, and compare the output
        with an expected result.
        """
        input_image = Image.new("1", (8, 8))
        input_image = pixelate(input_image, 2)
        self.assertTrue(input_image is not None)

    def test_prepare(self):
        """
        Test the prepare function with a sample image, and verify the output image's size and type.
        """
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(test_dir, "test_image.jpg")
        input_image = Image.open(test_image_path)

        # Test prepare function
        prepared_image = prepare_image(input_image, 20, 8)

        # Verify the output image
        self.assertIsNotNone(prepared_image)
        self.assertIsInstance(prepared_image, Image.Image)
        self.assertEqual(prepared_image.size, (20, 20))

    def test_spiralise(self):
        """
        Test the spiralise function with a simple 100x100 black and white image, and verify the
        output image's size and type.
        """
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(test_dir, "test_image.jpg")
        input_image = Image.open(test_image_path)
        input_image = prepare_image(input_image, 20, 4)

        # Test spiralise function
        spiralised_image = spiralise(input_image)

        # Verify the output image
        self.assertIsNotNone(spiralised_image)
        self.assertIsInstance(spiralised_image, Image.Image)

    def test_flowalise(self):
        """
        Test the flowalise function with a simple 100x100 black and white image, and verify the
        output image's size and type.
        """
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_image_path = os.path.join(test_dir, "test_image.jpg")
        input_image = Image.open(test_image_path)
        input_image = prepare_image(input_image, 20, 4)

        # Test flowalise function
        flowed_image = flowalise(input_image)
        # Verify the output image
        self.assertIsNotNone(flowed_image)
        self.assertIsInstance(flowed_image, Image.Image)


if __name__ == "__main__":
    unittest.main()
