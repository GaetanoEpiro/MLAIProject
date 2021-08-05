import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels

def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)

def generate_jigsaw_puzzle(names, beta, data_path):
  num_names = len(names)
  subset_indexes = sample(range(num_names), int(num_names*beta))

  for index in subset_indexes:
    fullpath = data_path + '/' + names[index]

    image = Image.open(fullpath)
    image.show()
    
    imgwidth, imgheight = image.size

    new_width = imgwidth + 1
    new_height = imgheight + 1 

    result = Image.new(image.mode, (new_width, new_height), (0, 0, 255))
    result.paste(image, (0, 0))

    imgwidth, imgheight = result.size
    
    x = imgwidth / 3
    y = imgheight / 3

    crops = []
  
    for i in range(0, imgwidth, int(x)):
      for j in range(0, imgwidth, int(y)):
        img = result.crop((j, i, int(j+x), int(i+y)))
        crops.append(img)

    #Read permutations from file
    num_perm = 30
    permutations = []

    with open('permutations_hamming_%d.txt' %(num_perm)) as f:
      lines = f.read().splitlines()

    for l in lines:
      permutations.append([int(i) for i in l.split()])

    #Select a permutation and reorder crops
    for permutation in permutations:
      permutate_img = [crops[i] for i in permutation]

      #Create the background for the new image
      new_image = Image.new('RGB', (imgwidth, imgheight))

      #Join crops
      k = 0
      for j in range(0, 3):
        for i in range(0, 3):
          new_image.paste(permutate_img[k], (i*int(x), j*int(y)))
          k += 1

class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None, beta=0.2):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.beta = beta

        generate_jigsaw_puzzle(names, beta, self.data_path)

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)



class TestDataset(Dataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index])


