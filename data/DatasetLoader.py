import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from random import sample

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

def generate_jigsaw_puzzle(permutations, image):
  
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

  #Select a random permutation and reorder crops
  label = np.random.randint(len(permutations)) + 1
  permutation = permutations[label - 1]

  permutate_img = [crops[i] for i in permutation]

  #Create the background for the new image
  new_image = Image.new('RGB', (imgwidth, imgheight))

  #Join crops
  k = 0
  for j in range(0, 3):
    for i in range(0, 3):
      new_image.paste(permutate_img[k], (i*int(x), j*int(y)))
      k += 1
  
  return new_image, label

def rotate_image(image):

  label = sample.randint(1, 3)

  switch = {
      1: image.transpose(Image.ROTATE_90),
      2: image.transpose(Image.ROTATE_180),
      3: image.transpose(Image.ROTATE_270)
  }

  image = switch.get(label)

  return image, label


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, img_transformer=None, beta_scrambled=0.2, beta_rotated=0.1, rotation=False):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.beta = beta_scrambled
        self.permutations = self.get_permutations()
        self.amount_scrambled = int(len(names) * beta_scrambled)
        self.amount_rotated = int(len(names) * beta_rotated)
        self.n_scrambled = 0
        self.rotation = rotation
        self.n_rotated = 0

    def __getitem__(self, index):
        
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        #Tasks:
        # - 0: classification
        # - 1: jigsaw puzzle (permutations)
        # - 2: rotation 
        # - 3: odd one out

        if self.n_scrambled < self.amount_scrambled:
          img, label = generate_jigsaw_puzzle(self.permutations, img)
          self.n_scrambled += 1
          img = self._image_transformer(img)

          #return image, image label, permutation label, task=permutation
          return img, int(self.labels[index]), label, int(1)

        if self.rotation==True and self.n_rotated < self.amount_rotated:
          img, label = rotate_image(self.permutations, img)
          self.n_rotated += 1
          img = self._image_transformer(img)

          #return image, image label, rotation label, task=rotation 
          return img, int(self.labels[index]), label, int(2)

        img = self._image_transformer(img)

        #return image, image label, permutate=false, task=classification
        return img, int(self.labels[index]), int(0), int(0)


    def __len__(self):
        return len(self.names)

    def get_permutations(self):
      permutations = []

      with open('permutations_hamming_30.txt') as f:
        lines = f.read().splitlines()

      for l in lines:
        permutations.append([int(i) for i in l.split()])

      return permutations



class TestDataset(Dataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index]), int(0)


