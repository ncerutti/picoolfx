from matplotlib import pyplot as plt
from PIL import Image
from picoolfx.picoolfx import pixelate, spiralise, flowalise
from picoolfx.utils import prepare_image


if __name__ == "__main__":
    image = Image.open("tests/test_image.jpg")
    image = prepare_image(image, 300, 16)
    image = flowalise(image)
    plt.imshow(image)
    plt.show()
