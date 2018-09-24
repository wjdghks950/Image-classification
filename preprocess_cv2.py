import cv2
import os
import random
from data_input_CIFAR10 import DataLoader

cifar10_datapath = '/home/jeonghwan/Desktop/CIFAR10_Jeonghwan/cifar-10-batches-py'

class ImgProcess():

    def __init__(self, datapath, edge_detect=False):
        self.opt = {}
        self.opt['datapath'] = datapath
        self.opt['edge'] = edge_detect

    def grayscale(self, x):
        gray_img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        print('Before resizing:{}'.format(gray_img.shape))
        resized_img = cv2.resize(gray_img, None, fx=5.0, fy=5.0)
        original_img = cv2.resize(x, None, fx=5.0, fy=5.0)
        print('After resizing:{}'.format(resized_img.shape))

        cv2.imshow('original_image', original_img)
        cv2.imshow('gray_image', resized_img)

        cv2.waitKey(0)
        return gray_img

    def load_sample(self):
        batch_idx = random.randint(1, 6)
        features, labels = DataLoader(self.opt['datapath'], batch_idx)

        sample_idx = random.randint(0, len(features))

        if not(0 <= sample_idx < len(features)):
            print('{} samples in batch{}. {} is out of range.'.format(len(features), batch_idx, sample_idx))

        print('\nBatch #{}'.format(batch_idx))
        print('Sample index #{}'.format(sample_idx))

        sample_img = features[sample_idx]
        sample_lbl = labels[sample_idx]

        self.grayscale(sample_img)

        print('Displaying image...')


def main():
    display = ImgProcess(cifar10_datapath)
    display.load_sample()


if __name__ == '__main__':
    main()
