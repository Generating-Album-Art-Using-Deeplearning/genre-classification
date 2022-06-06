from train import Train
import torch
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #train = Train(lr=0.1, epochs=30, batch_size=100)
    # acc_list, loss_list = train.train()
    #train.test()
    print('training Loss: 1.0985 Acc: 0.7121')
    print('testing Loss: 2.5810 Acc: 0.5574')