import numpy as np
import json
import os.path
import logging

'''
mel-spectrogram 데이터들을 모아놓은 input.npy 파일과 input 데이터에 매핑되는 장르 코드(e.g GN1600)를 0부터의 정수로 변환한 output.npy 파일을 생성한다.
이후 get_input() 과 get_output() 함수를 통해 input 데이터와 output 데이터를 불러올 수 있다.
'''

class Kakao_arena_dataset():
    def __init__(self):
        # kakao arena dataset file path
        self.__melspectrogram_folder_path = '/home/dnclab/graduationProject/data/arena_mel/'
        self.__song_meta_file_path = '/home/dnclab/graduationProject/data/song_meta.json'
        self.__gnere_all_file_path = '/home/dnclab/graduationProject/data/genre_gn_all.json'
        
        # 장르 코드(e.g. GN1600)들을 0부터 정수로 매핑시킨 json 파일 (장르가 비어있는 데이터의 경우 254가 라벨링 되어있음)
        self.__labeled_genre_json_path = '/home/dnclab/graduationProject/data/labeled_genre.json'

        # 모델에 input/output 으로 활용할 파일 경로
        self.__input_file_path = '/home/dnclab/graduationProject/data/input.npy'
        self.__output_file_path= '/home/dnclab/graduationProject/data/output.npy'

        # size가 (48,1876)이 아닌 mel-spectrogram 들의 set
        self.different_size_melspectrogram_set = set()
    
    '''
    public functions
    '''

    def run(self, start_id=0, end_id=1):
        if not os.path.isfile(self.__input_file_path):
            self.__save_input_file(start_id, end_id)
        if not os.path.isfile(self.__output_file_path):
            self.__save_output_file(start_id, end_id)

    def get_input(self):
        return np.load(self.__input_file_path)

    def get_output(self):
        return np.load(self.__output_file_path)


    '''
    file generate & save (private)functions
    '''

    def __save_input_file(self, start_id, end_id):
        melspectrogram_list = []
        for song_id in range(start_id, end_id):
            file_path = self.__melspectrogram_folder_path + str(song_id//1000) + '/' + str(song_id) + '.npy'
            melspectrogram = np.load(file_path)
            if melspectrogram.shape == (48, 1876):
                melspectrogram_list.append(melspectrogram)
                continue
            else:
                logging.debug('DATA EXEPTION (Mel-Spectrogram): {} mel-spectrogram size is not (48, 1876)'.format(song_id))
                self.different_size_melspectrogram_set.add(song_id)
        song_array = np.stack(melspectrogram_list, axis=0)
        a = (1,)
        b = song_array.shape
        c = b+a
        song_array = song_array.reshape((1796,1,48,1876))
        # song_array = song_array.reshape((1796,48,1876))
        print(song_array.shape)
        self.__save_npy_file(self.__input_file_path, song_array)

    def __save_output_file(self, start_id, end_id):
        if not os.path.isfile(self.__labeled_genre_json_path):
            self.__save_labeled_genre()
        
        labeled_genre = self.__load_json_file(self.__labeled_genre_json_path)
        song_meta = self.__load_json_file(self.__song_meta_file_path)
        genre_num_data = []

        for song_id in range(start_id, end_id):
            song = song_meta[song_id]
            genre_raw = song['song_gn_gnr_basket']
            genre_num = 254

            if song['id'] in self.different_size_melspectrogram_set:
                logging.debug('DATA EXEPTION (Genre): {} is in different_size_melspectrogram, so not in label data'.format(song_id))
                continue

            try:
                genre_num = labeled_genre[genre_raw[0]]
            except:
                logging.debug('DATA EXEPTION (Genre): {} genre is empty or wrong data')
            
            genre_num_data.append(genre_num)

        self.__save_npy_file(self.__output_file_path, genre_num_data)

    def __save_labeled_genre(self):
        genre_all = self.__load_json_file(self.__gnere_all_file_path)

        genre_keys = list(genre_all.keys())
        genre_len = len(genre_keys)
        genre_dict = {}
        for i in range(genre_len):
            genre_dict[genre_keys[i]] = i

        self.__write_json_file(self.__labeled_genre_json_path, genre_dict)


    '''
    util (private)functions
    '''

    def __load_json_file(self, path):
        with open(path, 'r') as json_file:
            return json.load(json_file)

    def __write_json_file(self, path, dict):
        with open(path, 'w') as outfile:
            json.dump(dict, outfile)
    
    def __save_npy_file(self, path, list):
        np_array = np.array(list)
        np.save(path, np_array)


# for debug
if __name__ == '__main__':
    dataset = Kakao_arena_dataset()
    dataset.run(0, 1800)
    dataset.get_output()
    