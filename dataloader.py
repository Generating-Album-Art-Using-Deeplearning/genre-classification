from cv2 import transform
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from torchvision import transforms
import cv2
from yaml import load


class Kakao_arena_dataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        with open('./data/song_meta.json', 'r') as json_file:
            self.song_meta = json.load(json_file)
        with open('./data/labeled_genre.json', 'r') as json_file:
            self.labeled_genre = json.load(json_file)
        
    def __len__(self):
        if self.mode == 'test':
            return 18000
        return 162000

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx = idx + 162000
        x = torch.FloatTensor(self.__getsepctrogram(idx))
        try:
            gnr_code = self.song_meta[idx]['song_gn_gnr_basket'][0]
            label = [self.labeled_genre[gnr_code]]
        except:
            label = [253]
        y = torch.FloatTensor(label)
        return x, y

    def __getsepctrogram(self, idx):
        file_path = './data/arena_mel/{}/{}.npy'.format(str(idx//1000), idx)
        loaded = np.load(file_path)
        if loaded.shape != (48,1876):
            loaded = cv2.resize(loaded,(1876,48))
        return loaded
        
if __name__ == "__main__":
    dataset = Kakao_arena_dataset()
    print(len(dataset))
    it = iter(dataset)
    for i in range(10):
        tmp = next(it)
        print(i, tmp[1].shape)