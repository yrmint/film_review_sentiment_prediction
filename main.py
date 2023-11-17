import re

from os import system
from train_model import TrainModel

if __name__ == '__main__':
    system('pip install -r requirements.txt')
    test_folder = r'./Data/'

    TrainModel(test_folder)
