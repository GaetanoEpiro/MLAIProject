from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.DatasetLoader import Dataset, TestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]

available_datasets = pacs_datasets

def get_train_dataloader(args):

    dataset_list = args.source
    assert isinstance(dataset_list, list)

    datasets = []
    val_datasets = []
    img_transformer = get_train_transformers(args)
    val_trasformer = get_val_transformer(args)

    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', dname+'.txt'), args.val_size)

        train_dataset = Dataset(name_train, labels_train, args.path_dataset, img_transformer=img_transformer, beta_scrambled=args.beta_scrambled, beta_rotated=args.beta_rotated)
        datasets.append(train_dataset)

        val_dataset = TestDataset(name_val, labels_val, args.path_dataset, img_transformer=val_trasformer, beta_scrambled=args.beta_scrambled, beta_rotated=args.beta_rotated)
        val_datasets.append(val_dataset)

    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader, val_loader

def get_jigsaw_dataloader(args):

    #Only for DA
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
    img_tr = get_train_transformers(args)

    train_dataset = Dataset(names, labels, args.path_dataset, img_transformer=img_tr, beta_scrambled=args.beta_scrambled, beta_rotated=args.beta_rotated)
    dataset = ConcatDataset([train_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader

def get_val_dataloader(args):

    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', args.target+'.txt'))
    img_tr = get_val_transformer(args)

    val_dataset = TestDataset(names, labels, args.path_dataset, img_transformer=img_tr, beta_scrambled=args.beta_scrambled, beta_rotated=args.beta_rotated)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]

    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    if args.random_grayscale:
        img_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


def get_val_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)
