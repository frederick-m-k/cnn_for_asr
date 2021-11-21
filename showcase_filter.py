
from keras.datasets import mnist

from matplotlib import pyplot as plt
from numpy.core.numeric import convolve


def convolute(image, filter):
    new_map = []
    for row_count in range(len(image)-len(filter)):
        new_map.append([])
        for col_count in range(len(image[row_count]-len(filter))):
            conv_value = 0
            for i in range(len(filter)):
                for j in range(len(filter[i])):
                    conv_value = conv_value + \
                        (image[row_count+i][col_count-j] * filter[i][j])
            new_map[row_count].append(conv_value)
    return new_map


(train_x, train_y), (test_x, test_y) = mnist.load_data()

# plot it
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_x[i], cmap=plt.get_cmap("gray"))
plt.show()

five_image = train_x[0]

filter = [
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
]

plt.imshow(five_image, cmap=plt.get_cmap("gray"))
plt.show()

result = convolute(five_image, filter)

plt.subplot(220+1)
plt.imshow(five_image, cmap=plt.get_cmap("gray"))
plt.subplot(220+1+1)
plt.imshow(result, cmap=plt.get_cmap("gray"))
plt.show()
