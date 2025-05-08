import numpy as np
import os
from PIL import Image
import random
import matplotlib.pyplot as plt

class FunnyImageNet:
    def __init__(self, directory):
        dataset = {}
        for s in os.listdir(directory):
            images = []
            for img in os.listdir(os.path.join(directory, s)):
                imgdata = Image.open(os.path.join(directory, s, img))
                imgdata = np.array(imgdata)
                imgdata = imgdata.astype(float)
                imgdata = imgdata / 255
                images.append(imgdata)
            dataset[s] = np.stack(images)
        self.dataset = dataset
        self.mapping = {
            s : idx for idx, s in enumerate(sorted(dataset.keys()))
        }
        self.inverse_mapping = {
            idx : s for idx, s in enumerate(sorted(dataset.keys()))
        }
        

    def label_to_id(self, label: str):
        return self.mapping[label]

    def id_to_label(self, id: int):
        return self.inverse_mapping[id]

    # returns x_train, y_train, x_test, y_test
    def sets(self, split=0.2) -> tuple:
        x_train, y_train = [], []
        x_test, y_test = [], []
        dataset = self.dataset
        
        for class_name, images in dataset.items():
            label = self.mapping[class_name]
            n_total = images.shape[0]
            
            indices = list(range(n_total))
            random.shuffle(indices)
            
            n_train = int(n_total * (1 - split))
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            x_train.append(images[train_idx])
            y_train.extend([label] * len(train_idx))
            
            x_test.append(images[test_idx])
            y_test.extend([label] * len(test_idx))
        
        x_train = np.concatenate(x_train)
        y_train = np.array(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.array(y_test)

        train_indices = list(range(len(x_train)))
        test_indices = list(range(len(x_test)))
        
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        x_test = x_test[test_indices]
        y_test = y_test[test_indices]
        
        
        return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    fn = FunnyImageNet('./data')
    print(fn.mapping)
    print(fn.inverse_mapping)
    x_train, y_train, x_test, y_test = fn.sets()
    print(y_train, y_test)
    plt.imshow(x_train[70])
    print(y_train[70], fn.id_to_label(y_train[70]))    #print(x_train, x_test)
    plt.show();
