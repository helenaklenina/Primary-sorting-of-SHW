import numpy as np
import torch
import pickle
import torch.onnx
import random

from pathlib import Path
from tqdm import tqdm, tqdm_notebook

import torch.nn as nn

from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from matplotlib import colors, pyplot as plt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from waste_dataset import WasteDataset
from trainer import train


DEVICE = torch.device("cuda")

def retrain_model():

    model_res50 = models.wide_resnet50_2(pretrained=True)

    TRAIN_DIR = Path('./waste_data/TRAIN')
    TEST_DIR = Path('./waste_data/TEST')

    train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
    test_files = sorted(list(TEST_DIR.rglob('*.jpg')))

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                            stratify=train_val_labels)

    train_dataset = WasteDataset(train_files, mode='train')

    val_dataset = WasteDataset(val_files, mode='val')

    test_dataset = WasteDataset(test_files, mode='test')

    n_classes = len(np.unique(train_val_labels))
    num_features = 2048
    # layears_freeze = 5

    # for param in full_model.parameters():
    #     param.requires_grad = False
        
    model_res50.fc = nn.Linear(num_features, n_classes)

    model_res50 = model_res50.to(DEVICE)

    model, history = train(train_dataset, val_dataset, model=model_res50, epochs=10, batch_size=16)


def set_seed(seed):

  # Устанавливаем начальное число для генерации простых чисел
  torch.manual_seed(seed)

  # Устанавливаем начальное число для генерации простых чисел
#   torch.use_deterministic_algorithms(True)

  # Устанавливаем начальное число для генерации простых чисел
#   torch.backends.cudnn.deterministic = True

  # Устанавливаем начальное число для генерации простых чисел
#   torch.backends.cudnn.baenchmark = False

  # Устанавливаем начальное число для генерации простых чисел
  np.random.seed(seed)

  # Устанавливаем начальное число для генерации простых чисел
  random.seed(seed)

  # Значение os.environ известно как объект мэппинга (сопоставления), 
  # который работает со словарем переменных пользовательской среды. 
  os.environ['PYTHONHASHSEED'] = str(seed)

def save_model(model):
    PATH = Path.joinpath('./', "my_resnet.onnx")
    dummy_input = Variable(torch.randn(1, 3, 224, 224)).to(DEVICE)
    torch.onnx.export(model, dummy_input, PATH)
