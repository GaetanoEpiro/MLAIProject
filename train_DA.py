import argparse

import torch
from torch import nn
from torch.nn import functional as F
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--path_dataset", default="/content/MLAIProject/", help="Path where the dataset is located")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--random_grayscale", default=0.1, type=float,help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), default="resnet18", help="Which network to use")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--train_all", type=bool, default=True, help="If true, all network weights will be trained")

    # tensorboard logger
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

    #Jigsaw Puzzle
    parser.add_argument("--beta", type=float, default=0.2, help="Percentage of images used to solve the Jigsaw puzzle")

    return parser.parse_args()


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = model_factory.get_network(args.network)(classes=args.n_classes)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_val_dataloader(args)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)

        self.n_classes = args.n_classes

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, (data, class_l) in enumerate(self.source_loader):

            data, class_l = data.to(self.device), class_l.to(self.device)

            self.optimizer.zero_grad()

            class_logit = self.model(data)
            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)

            loss = class_loss

            loss.backward()

            self.optimizer.step()


            self.logger.log(it, len(self.source_loader),
                            {"Class Loss ": class_loss.item()},
                            {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item()},
                            data.shape[0])
            del loss, class_loss,class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = self.do_test(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"Classification Accuracy": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, (data, class_l) in enumerate(loader):
            data, class_l = data.to(self.device), class_l.to(self.device)
            class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        for self.current_epoch in range(self.args.epochs):
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
            self.scheduler.step()

        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
