import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from data import data_helper
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import itertools
import torch.nn.functional as func

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
    parser.add_argument("--beta_scrambled", type=float, default=0.01, help="Percentage of images used to solve the Jigsaw puzzle")
    parser.add_argument("--alpha_target", type=float, default=0.5, help="Target Jigen loss weight during training")
    parser.add_argument("--alpha_source", type=float, default=0.5, help="Source Jigen loss weight during training")
    parser.add_argument("--target_loss_wt", type=float, default=0.1, help="Target class loss weight during training")

    parser.add_argument("--rotation", type=bool, default=False, help="")
    parser.add_argument("--beta_rotated", type=float, default=0.1, help="Percentage of rotated images")
    parser.add_argument("--alpha_rotation_target", type=float, default=0.5, help="Target rotation loss weight during training")
    parser.add_argument("--alpha_rotation_source", type=float, default=0.5, help="Source rotation loss weight during training")

    parser.add_argument("--odd_one_out", type=bool, default=False, help="")
    parser.add_argument("--beta_odd", type=float, default=0.1, help="")
    parser.add_argument("--alpha_odd_target", type=float, default=0.5, help="")
    parser.add_argument("--alpha_odd_source", type=float, default=0.5, help="")

    return parser.parse_args()

def entropy_loss(x):
    return torch.sum(-func.softmax(x, 1) * func.log_softmax(x, 1), 1).mean()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = model_factory.get_network(args.network)(classes=args.n_classes, jigsaw_classes=31, rotation_classes=4, odd_classes=9)
        self.model = model.to(device)

        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args)
        self.target_loader = data_helper.get_val_dataloader(args)
        self.target_jigsaw = data_helper.get_jigsaw_dataloader(args)

        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))

        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all)

        self.n_classes = args.n_classes

        self.nTasks = 2
        print(args.rotation)
        if args.rotation == True:
            self.nTasks += 1
        if args.odd_one_out == True:
            self.nTasks += 1

        print("N of tasks: " + str(self.nTasks))

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for it, ((data, class_l, jigsaw_label, task_type), (data_target_jigsaw, class_target_jigsaw, target_jigsaw_label, task_type_target)) in enumerate(zip(self.source_loader, itertools.cycle(self.target_jigsaw))):
        #for it, (data, class_l) in enumerate(self.source_loader):

            rotation_loss = 0
            rotation_target_loss = 0
            odd_loss = 0
            odd_target_loss = 0
            rotation_pred = 0
            odd_pred = 0

            data, class_l, jigsaw_label, task_type = data.to(self.device), class_l.to(self.device), jigsaw_label.to(self.device), task_type.to(self.device)          
            data_target_jigsaw, class_target_jigsaw, target_jigsaw_label, task_type_target = data_target_jigsaw.to(self.device), class_target_jigsaw.to(self.device), target_jigsaw_label.to(self.device), task_type_target.to(self.device)

            self.optimizer.zero_grad()

            #Source logit
            class_logit, jigsaw_logit, rotation_logit, odd_logit = self.model(data)
            
            #Source loss only for classification
            class_loss = criterion(class_logit[task_type==0], class_l[task_type==0])
            
            #Jigsaw loss if the task is classification or "puzzle"
            jigsaw_loss = criterion(jigsaw_logit[(task_type==0) | (task_type==1)], jigsaw_label[(task_type==0) | (task_type==1)])

            #Target logit
            class_logit_target, jigsaw_logit_target, rotation_logit_target, odd_logit_target = self.model(data_target_jigsaw) 

            #Target loss if the task is classification 
            class_target_loss = entropy_loss(class_logit_target[task_type_target==0])

            #Target jigsaw loss if the task is classification or "puzzle"
            jigsaw_target_loss = criterion(jigsaw_logit_target[(task_type_target==0) | (task_type_target==1)], target_jigsaw_label[(task_type_target==0) | (task_type_target==1)])

            if self.args.rotation == True:
                #Rotation loss if the task is classification of "rotation"
                rotation_loss = criterion(jigsaw_logit[(task_type==0) | (task_type==2)], jigsaw_label[(task_type==0) | (task_type==2)])

                #Target rotation loss if the target is classification or "rotation"
                rotation_target_loss = criterion(rotation_logit_target[(task_type_target==0) | (task_type_target==2)], target_jigsaw_label[(task_type_target==0)| (task_type_target==2)])

                _, rotation_pred = rotation_logit.max(dim=1)

            if self.args.odd_one_out == True:
                #Odd one out loss if the task is classification of "rotation"
                odd_loss = criterion(jigsaw_logit[(task_type==0) | (task_type==3)], jigsaw_label[(task_type==0) | (task_type==3)])

                #Target rotation loss if the target is classification or "rotation"
                odd_target_loss = criterion(odd_logit_target[(task_type_target==0) | (task_type_target==3)], target_jigsaw_label[(task_type_target==0)| (task_type_target==3)])

                _, odd_pred = odd_logit.max(dim=1)

            _, cls_pred = class_logit.max(dim=1)
            _, jigsaw_pred = jigsaw_logit.max(dim=1)

            class_loss = class_loss + class_target_loss * self.args.target_loss_wt
            jigsaw_loss = jigsaw_target_loss * self.args.alpha_target + jigsaw_loss * self.args.alpha_source
            rotation_loss = rotation_target_loss * self.args.alpha_rotation_target + rotation_loss * self.args.alpha_rotation_source
            odd_loss = odd_target_loss * self.args.alpha_odd_target + odd_loss * self.args.alpha_odd_source

            loss = class_loss + jigsaw_loss + rotation_loss + odd_loss

            loss.backward()

            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"Class Loss ": class_loss.item()},
                            {"Class Accuracy ": torch.sum(cls_pred == class_l.data).item()},
                            data.shape[0])

            self.logger.log(it, len(self.source_loader),
                            {"Jigsaw Loss ": jigsaw_loss.item()},
                            {"Jigsaw Accuracy ": torch.sum(jigsaw_pred == jigsaw_label.data).item()},
                            data.shape[0])

            if self.args.rotation == True:
                self.logger.log(it, len(self.source_loader),
                                {"Rotation Loss ": rotation_loss.item()},
                                {"Rotation Accuracy ": torch.sum(rotation_pred == jigsaw_label.data).item()},
                                data.shape[0])

            if self.args.odd_one_out == True:
                self.logger.log(it, len(self.source_loader),
                                {"Odd one out Loss ": odd_loss.item()},
                                {"Odd one out Accuracy ": torch.sum(odd_pred == jigsaw_label.data).item()},
                                data.shape[0])

            del class_logit, jigsaw_loss, jigsaw_logit, class_logit_target, jigsaw_logit_target, class_target_loss, jigsaw_target_loss, class_loss, loss
            del rotation_logit, rotation_loss, rotation_target_loss
            del odd_loss, odd_logit, odd_target_loss, rotation_pred, odd_pred

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct, jigsaw_correct, rotation_correct, odd_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                jigsaw_acc = float(jigsaw_correct) / total
                rotation_acc = 0
                odd_acc = 0

                if self.args.rotation == True:
                    rotation_acc = float(rotation_correct) / total

                if self.args.odd_one_out == True:
                    odd_acc = float(odd_correct) / total

                accuracy = (class_acc + jigsaw_acc + rotation_acc + odd_acc) / self.nTasks

                self.logger.log_test(phase, {"Classification Accuracy": accuracy})
                self.results[phase][self.current_epoch] = accuracy

    def do_test(self, loader):
        class_correct = 0
        jigsaw_correct = 0
        rotation_correct = 0
        odd_correct = 0

        for it, (data, class_l, jigsaw_label, task_type) in enumerate(loader):
            data, class_l, jigsaw_label, task_type = data.to(self.device), class_l.to(self.device), jigsaw_label.to(self.device), task_type.to(self.device)
            
            class_logit, jigsaw_logit, rotation_logit, odd_logit = self.model(data)
            
            _, cls_pred = class_logit.max(dim=1)
            _, jigsaw_pred = jigsaw_logit.max(dim=1)

            if self.args.rotation == True:
                _, rotation_pred = rotation_logit.max(dim=1)
                rotation_correct += torch.sum(rotation_pred == jigsaw_label.data)
    
            if self.args.odd_one_out == True:
                _, odd_pred = odd_logit.max(dim=1)
                odd_correct += torch.sum(odd_pred == jigsaw_label.data)

            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jigsaw_pred == jigsaw_label.data)

        return class_correct, jigsaw_correct, rotation_correct, odd_correct

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
