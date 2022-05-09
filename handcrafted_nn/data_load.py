import numpy as np
import struct
from handcrafted_nn.pure_data import pure_data
import os
def read_mnist(image, label,train = True):
    if train:
        img_file_size = img_file_size = 47040016
        label_file_size = 60008
    else:
        img_file_size = img_file_size = 7840016
        label_file_size = 10008
    img_file_size = str(img_file_size - 16) + 'B'
    magic, nImages, nRows, nColumns=struct.unpack_from('>4I', open(image, 'rb').read(), 0)
    imgs = struct.unpack_from('>' + img_file_size, open(image, 'rb').read(), struct.calcsize('>IIII'))
    imgs = np.array(imgs).astype(np.uint8).reshape(nImages, 1, nRows, nColumns)

    label_file_size = str(label_file_size - 8) + 'B'
    magic, nlabels=struct.unpack_from('>2I', open(label, 'rb').read(), 0)
    labels = struct.unpack_from('>' + label_file_size, open(label, 'rb').read(), struct.calcsize('>II'))
    labels = np.array(labels).astype(np.int64)

    return imgs, labels

def img_resize(imgs, size):
    imgs_resize = []
    _,_,old_row, old_col = imgs.shape
    for img in imgs:
        img = img[0,:,:]
        new_img= np.expand_dims(np.pad(img, (((size[0]-img.shape[0])//2,(size[0]-img.shape[0])//2), ((size[1]-img.shape[1])//2,(size[1]-img.shape[1])//2)), mode='constant', constant_values=((0,0),(0,0))), 0)
        imgs_resize.append(new_img)
    imgs_resize = np.array(imgs_resize)
    return imgs_resize

def one_hot(labels, num_class):
    one_hot_label = []
    for label in labels:
        one_hot = np.zeros(num_class)
        one_hot[label] = 1
        one_hot_label.append(one_hot)
    return np.array(one_hot_label)


def dataloader(image, label , batch_size = 1, shuffle=False):
    if shuffle:
        # Implement in the future
        pass
    else:
        dataset = []
        for i in range(len(label)//batch_size):
            # print(np.array(image[i:i+batch_size,:,:,:], label[i:i+batch_size]))
            dataset.append([pure_data(image[i:i+batch_size,:,:,:]/255), pure_data(label[i:i+batch_size])])
        return dataset

if __name__ == "__main__":
    img, label = read_mnist("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte")
    img = img_resize(img,(32,32))
    label = one_hot(label,10)
    dataset = dataloader(img, label, batch_size=1000)
    for i, (x, y) in enumerate(dataset):
        print(x)
        print("第%d個batch(%s, %s)" %(i, x.shape, y.shape))