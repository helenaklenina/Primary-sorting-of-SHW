import torch
import pickle


from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from PIL import Image

from matplotlib import colors, pyplot as plt

DEVICE = torch.device("cuda")

# разные режимы датасета 
DATA_MODES = ['train', 'test', 'val']

class WasteDataset(Dataset):
    """ 
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """

    def __init__(self, files, mode):
      super().__init__()
      self.files = sorted(files)
      self.mode = mode

      if self.mode not in DATA_MODES:
        print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
        raise NameError

      self.len_ = len(self.files)
      
      self.label_encoder = LabelEncoder()

      if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_
    
    def load_sample(self, file):
      """ Загрузка изображения """
      image = Image.open(file).convert('RGB')
      image.load()
      return image

    def _prepare_sample(self, image, rescale_size):
        """ Корретировка размера изображения """
        image = image.resize((rescale_size, rescale_size))
        return image

    def __getitem__(self, index):
      transform = {
          'train': transforms.Compose([
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ]),
          'test': transforms.Compose([
              transforms.CenterCrop(224),
              transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
      }

      x = self.load_sample(self.files[index])
      x = self._prepare_sample(x)

      if self.mode == 'test':
        x = transform['test'](x)
      else:
        x = transform['train'](x)

      if self.mode == 'test':
        return x
      else:
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        return x, y