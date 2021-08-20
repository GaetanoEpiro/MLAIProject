import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from random import sample
import random
import math 
from models.style_transfer_model import Model

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
    
  x = math.ceil(imgwidth / 3)
  y = math.ceil(imgheight / 3)

  crops = []
  
  for i in range(0, imgwidth, x):
    for j in range(0, imgheight, y):
      crop = image.crop((j, i, j+x, i+y))
      crops.append(crop)

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
      new_image.paste(permutate_img[k], (i*x, j*y))
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

def generate_odd_one_out_image(image, names, path):

  #Divide the image into crops
  image_crops = []

  imgwidth, imgheight = image.size

  x = math.ceil(imgwidth / 3)
  y = math.ceil(imgheight / 3)

  for i in range(0, imgwidth, x):
    for j in range(0, imgheight, y):
      crop = image.crop((j, i, j+x, i+y))
      image_crops.append(crop)

  #Select a random image
  index = random.randint(0, len(names))

  framename = path + '/' + names[index]
  random_image = Image.open(framename).convert('RGB')

  crops = []

  #Divide the random image into crops
  for i in range(0, imgwidth, x):
    for j in range(0, imgheight, y):
      crop = random_image.crop((j, i, j+x, i+y))
      crops.append(crop)

  #Select a random crop
  pos = random.randint(0, 8)

  #Replace the crop inside the original image
  image_crops[pos] = crops[pos]

  new_image = Image.new('RGB', (imgwidth, imgheight))

  k = 0
  for j in range(0, 3):
    for i in range(0, 3):
      new_image.paste(image_crops[k], (i*x, j*y))
      k += 1

  return new_image, pos

class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, type_domain, target, jigsaw_style_transfer, img_transformer=None, beta_scrambled=0.2, beta_rotated=0.1, beta_odd=0.1, rotation=False, odd=False):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.beta = beta_scrambled
        self.permutations = self.get_permutations()
        self.amount_scrambled = int(len(names) * beta_scrambled)
        self.amount_rotated = int(len(names) * beta_rotated)
        self.amount_odd = int(len(names) * beta_odd)
        self.n_scrambled = 0
        self.n_rotated = 0
        self.n_odd = 0
        self.rotation = rotation
        self.odd = odd


        # Style Transfer 
        self.type_domain = type_domain
        self.target = target
        self.jigsaw_style_transfer = jigsaw_style_transfer
        self.type_style_transfer = "All"
        self.style_model = self.find_model()



        self.nTasks = 2
        if rotation==True:
          self.nTasks += 1
        if odd==True:
          self.nTasks += 1

    
    def find_model(self): 
      model = Model()
      name = ""

      #if self.type_domain == "DG": 
        #name_model = "style_transfer_model_" + self.target
      #elif self.type_domain == "DA": 
        #name_model = "style_transfer_model_" + "All"
      
      name_model = "style_transfer_model"
      model_state = torch.load("models/pretrained/" + name_model, map_location=lambda storage, loc: storage)
      model.load_state_dict(model_state, strict=False)
      model = model.to(self.device)



      return model


    def denorm(self, tensor):
      std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(self.device)
      mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(self.device)
           
      res = torch.clamp(tensor * std + mean, 0, 1)
      return res




    def change_style(self, img, index): 

      # it can be: photo, cartoon, sketch, art painting 
      current_style = (self.names[index].split('/'))[2]

      # "/content/MLAIProject" + "PACS" + "kfold"
      current_path = self.data_path + '/' + self.names[index].split("/")[0] + '/' + self.names[index].split("/")[1]

      # art_painting, cartoon, photo, sketch
      domains = os.listdir(current_path)

      if self.type_domain == "DG":
        # DG doesnt use the target  
        domains.remove(current_style)
        domains.remove(self.target)

      elif self.type_style_transfer == "All": 
        domains.remove(current_style)

      else: 
        domains = [self.target]
      
      styles_chosen = []

      # find 9 new styles for our img 
      while len(styles_chosen) != 9: 
        # randomly choose one domain
        img_domain = domains[np.random.randint(len(domains))]
        new_path = current_path + "/" + img_domain

        # randomly choose class of the image 
        img_class = os.listdir(new_path)[np.random.randint(len(os.listdir(new_path)))]
        new_path = new_path + "/" + img_class

        #randomly choose one img 
        img_chosen = os.listdir(new_path)[np.random.randint(len(os.listdir(new_path)))]
        new_path = new_path + "/" + img_chosen

        new_img_chosen = self._image_transformer(Image.open(new_path).covert("RGB"))
        styles_chosen.append(new_img_chosen)
      
      

      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      trans = transforms.Compose([transforms.ToTensor(), normalize])
        
      img_size = img.size[0]
      length = img_size / 3


      crops = []

      for index, style in enumerate(styles_chosen): 
        for i in range(3): 
          for j in range(3): 

            style_t = trans(style).unsqueeze(0).to(self.device)
            img_t = trans(img.crop([i * length, j * length, (i + 1) * length, (j + 1) * length])).unsqueeze(0).to(self.device)

            with torch.no_grad():
              out = self.style_model.generate(img_t, style_t, 1)

            out = self.denorm(out)
            out = torch.squeeze(out)
            crops.append(out)
            

            # transform out in an image in order to do the same crops we did in generate_jigsaw_puzzle

      imgwidth, imgheight = image.size
    
      x = math.ceil(imgwidth / 3)
      y = math.ceil(imgheight / 3)
      new_image = Image.new('RGB', (imgwidth, imgheight))

      #Join crops
      k = 0
      for j in range(0, 3):
        for i in range(0, 3):
          new_image.paste(crops[k], (i*x, j*y))
          k += 1
  

      new_image = transforms.ToPILImage(mode="RGB")(new_image)
      new_image = transforms.Resize([img_size, img_size])(new_image)


      # crop the image and then do the jigsaw puzzle 
      # (same as we did in generate_jigsaw_puzzle)
      label = np.random.randint(len(self.permutations)) + 1
      permutation = self.permutations[label - 1]

      crops = []
  
      for i in range(0, imgwidth, x):
        for j in range(0, imgheight, y):
          crop = new_image.crop((j, i, j+x, i+y))
          crops.append(crop)

      permutate_img = [crops[i] for i in permutation]

      #Create the background for the new image
      n_image = Image.new('RGB', (imgwidth, imgheight))

      #Join crops
      k = 0
      for j in range(0, 3):
        for i in range(0, 3):
          n_image.paste(permutate_img[k], (i*x, j*y))
          k += 1
      
      return n_image, label


      


    
    
    def __getitem__(self, index):
        
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')

        task = random.randint(0, self.nTasks)

        #Tasks:
        # - 0: classification
        # - 1: jigsaw puzzle (permutations)
        # - 2: rotation 
        # - 3: odd one out

        if task==1 and self.n_scrambled < self.amount_scrambled:
          # if self.jigsaw_style_transfer == true then i call function change_style(img, index)
          img, label = generate_jigsaw_puzzle(self.permutations, img)
          self.n_scrambled += 1
          img = self._image_transformer(img)

          #return image, image label, permutation label, task=permutation
          return img, int(self.labels[index]), label, int(1)

        if task==2 and self.rotation==True and self.n_rotated < self.amount_rotated:
          img, label = rotate_image(img)
          self.n_rotated += 1
          img = self._image_transformer(img)

          #return image, image label, rotation label, task=rotation 
          return img, int(self.labels[index]), label, int(2)

        if task==3 and self.odd==True and self.n_odd < self.amount_odd:
          img, label = generate_odd_one_out_image(img, self.names, self.data_path)
          self.n_odd += 1
          img = self._image_transformer(img)

          #return image, image label, rotation label, task=odd one out
          return img, int(self.labels[index]), label, int(3)

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

        return img, int(self.labels[index]), int(0), int(0)


