import matplotlib.pyplot as plt
import numpy as np

with open("data.txt", "r+") as f:
    data = f.readlines();
    arr = []

    for line in data:
        m = []
        for i, c in enumerate(line[:-1]):
            m.append(4 * (ord(c) - 33))
        arr.append(m)
    np_arr = np.array(arr)
    np_data = np_arr.reshape((80, 80, 4))
    np_data = np_data[:, :, 0:-1]

    #np_data[:, :, 0] = np.roll(np_data[:, :, 0], -1, 1)
    #np_data[:, :, 0] = np.roll(np_data[:, :, 0], 1, 0)

    plt.imshow(np_data)
    plt.show()
