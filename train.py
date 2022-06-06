from torch import nn, optim
import torch
import copy
import logging
from threading import Thread
from torch.utils.data import DataLoader, random_split
import threading
from model import W2VModel
import torch.nn.functional as F
from dataloader import Kakao_arena_dataset
import numpy as np
import time

class Train:
    def __init__(self, lr, epochs, batch_size):
        self.train_dataset = Kakao_arena_dataset('train')
        self.test_dataset = Kakao_arena_dataset('test')
        '''
        self.train_size = int(len(self.dataset)*0.8)
        self.validation_size = int(len(self.dataset)*0.1)
        self.test_size = int(len(self.dataset)*0.1)
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(self.dataset, [self.train_size, self.validation_size, self.test_size])
        '''
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.data_list = []
        self.acc_list = []
        self.loss_list = []

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = W2VModel(
            input_dim = 48,
            hidden_dim=513,
            stride=[5, 4, 2, 2, 2],
            filter_size=[10, 8, 4, 2, 2],
            padding=[2, 2, 2, 2, 1])
        model.load_state_dict(torch.load('./model_weights.pt'))
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr)

        dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        for e in range(self.epochs):
            running_loss = 0
            running_corrects = 0
            epoch_loss = 0
            epoch_acc = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                labels=labels.type(torch.cuda.LongTensor)
                loss = criterion(outputs, labels.squeeze(dim=1))
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                corrects = 0
                for i in range(self.batch_size):
                    if labels.data[i] == preds[i]:
                        corrects+=1
                running_corrects += corrects

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = int(running_corrects) / (len(dataloader)*self.batch_size)
            self.loss_list.append(epoch_loss)
            self.acc_list.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format("training", epoch_loss, epoch_acc))

        # torch.save(model, './model.pt')
        torch.save(model.state_dict(), './model_weights.pt')
        return self.acc_list, self.loss_list
    
    def test(self):
        PATH = './model_weights.pt'
        corrects = 0
        test_loss = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = W2VModel(
            input_dim = 48,
            hidden_dim=513,
            stride=[5, 4, 2, 2, 2],
            filter_size=[10, 8, 4, 2, 2],
            padding=[2, 2, 2, 2, 1])
        model.load_state_dict(torch.load(PATH))
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=True)
        running_corrects = 0
        for inputs, labels in dataloader:
            loss = 0
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            labels=labels.type(torch.cuda.LongTensor)
            loss = criterion(outputs, labels.squeeze(dim=1))
            _, preds = torch.max(outputs, 1)
            test_loss += loss.item() * inputs.size(0)

            corrects = 0
            for i in range(self.batch_size):
                if labels.data[i] == preds[i]:
                    corrects+=1
            running_corrects += corrects

        acc = int(running_corrects) / len(dataloader.dataset)
        avg_loss = test_loss / len(dataloader.dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("testing", avg_loss, acc))
        return acc, avg_loss

if  __name__ == '__main__':
    train = Train(lr=0.01, epochs=2, batch_size=10)
    train.train()