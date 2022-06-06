import torch
import numpy as np
import json

class Test_one_song():
    def __init__(self, id):
        self.test_song_id = id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load("./model.pt", map_location=self.device)
        with open('./data/song_meta.json', 'r') as json_file:
            self.song_meta = json.load(json_file)
        with open('./data/labeled_genre.json', 'r') as json_file:
            self.labeled_genre = json.load(json_file)
        with open('./data/genre_gn_all.json', 'r') as json_file:
            self.genre_all = json.load(json_file)

    def test(self):
        with torch.no_grad():
            self.model.eval()
            self.model.load_state_dict(torch.load('./model_weights.pt'))
            melspectrogram = np.load('./data/{}.npy'.format(self.test_song_id))
            melspectrogram = melspectrogram.reshape((1,48,1876))
            inputs = torch.FloatTensor(melspectrogram).to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            song_name, artist = self.get_song_meta()
            answer = self.get_answer()
            predicted = self.get_predicted(preds)

            print('answer genre: {}     | predicted genre : {}'.format(answer, predicted))

    def get_answer(self):
        genre_code = self.song_meta[self.test_song_id]['song_gn_gnr_basket'][0]
        genre = self.genre_all[genre_code]
        return '{}({})'.format(genre_code, genre)

    def get_predicted(self, preds):
        key_list = list(self.labeled_genre.keys())
        genre_code = key_list[preds.item()]
        genre = self.genre_all[genre_code]
        return '{}({})'.format(genre_code, genre)

    def get_song_meta(self):
        song_name = self.song_meta[self.test_song_id]['song_name']
        artist = self.song_meta[self.test_song_id]['artist_name_basket']
        return song_name, artist

if __name__ == '__main__':
    test = Test_one_song(210647)
    test.test()

