from matplotlib import pyplot as plt
from PIL import Image
from picoolfx.picoolfx import pixelate, prepare_image, spiralise, flowalise


if __name__ == "__main__":
    image = Image.open("tests/test_image.jpg")
    image = prepare_image(image, 300, 16)
    image = spiralise(image)
    plt.imshow(image)
    plt.show()
