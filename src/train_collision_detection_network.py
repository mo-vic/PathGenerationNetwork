import os
import sys
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.backends import cudnn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from logger import Logger

from torch.nn import BCELoss


class CollisionDetectionNetwork(nn.Module):
    def __init__(self, input_shape):
        super(CollisionDetectionNetwork, self).__init__()

        in_node = input_shape[1]
        self.layer1 = nn.Linear(in_node, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.layer4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.layer5 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x))) + x
        x = F.relu(self.bn3(self.layer3(x))) + x
        x = F.relu(self.bn4(self.layer4(x))) + x

        return torch.sigmoid(self.layer5(x))


class CollisionDataset(Dataset):
    def __init__(self, datafile, factor=1):
        super(CollisionDataset, self).__init__()

        self.factor = factor

        if isinstance(datafile, str):
            if not os.path.exists(datafile):
                print("{0} does not exist!".format(datafile))
                raise FileNotFoundError
        else:
            print("'datafile' should have type of str, found {0}".format(type(datafile)))
            raise TypeError

        self.data = np.load(datafile)

    def __getitem__(self, index):
        sample = self.data[index % len(self.data)]
        theta1, theta2, bbox_width, bbox_height, label = sample

        data, label = [theta1 / 180 * np.pi, theta2 / 180 * np.pi, bbox_width, bbox_height], label

        return torch.tensor(data, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        return len(self.data) * self.factor


def load_dataset(train_datafile, val_datafile, batch_size, use_gpu, num_workers):
    trainset = CollisionDataset(train_datafile, factor=1)
    trainfullset = CollisionDataset(train_datafile, factor=1)
    testset = CollisionDataset(val_datafile, factor=1)
    input_shape = (1, 4)

    trainloader = DataLoader(trainset, batch_size, True, num_workers=num_workers, pin_memory=use_gpu, drop_last=True)
    trainfullloader = DataLoader(trainfullset, batch_size, False, num_workers=num_workers, pin_memory=use_gpu, drop_last=False)
    testloader = DataLoader(testset, batch_size, False, num_workers=num_workers, pin_memory=use_gpu, drop_last=False)

    return trainloader, trainfullloader, testloader, input_shape


def build_model(input_shape):
    model = CollisionDetectionNetwork(input_shape)

    return model


def train(model, dataloader, criterion, optimizer, use_gpu, writer, epoch):
    model.train()

    all_acc = []
    all_loss = []

    for idx, (data, labels) in tqdm(enumerate(dataloader), desc="Training Epoch {}".format(epoch)):
        optimizer.zero_grad()
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs = model(data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())
        acc = (torch.ge(outputs.data, 0.5) == labels.data).double().mean()
        all_acc.append(acc.item())

        writer.add_scalar("loss", loss.item(), global_step=epoch * len(dataloader) + idx)
        writer.add_scalar("acc", acc.item(), global_step=epoch * len(dataloader) + idx)

    print("Epoch {}: total trainset loss: {}, global trainset accuracy:{}".format(epoch, np.mean(all_loss),
                                                                                  np.mean(all_acc)))


def eval(model, dataloader, criterion, scheduler, use_gpu, writer, log_dir, epoch, best_acc):
    model.eval()

    all_acc = []
    all_loss = []

    with torch.no_grad():
        for idx, (data, labels) in tqdm(enumerate(dataloader), desc="Evaluating Epoch {}".format(epoch)):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            loss = criterion(outputs, labels)

            all_loss.extend(loss.cpu().numpy().tolist())
            acc = (torch.ge(outputs.data, 0.5) == labels.data).double().cpu().numpy().tolist()
            all_acc.extend(acc)

        val_loss = np.mean(all_loss)
        val_acc = np.mean(all_acc)
        writer.add_scalar("val_loss", val_loss, global_step=epoch)
        writer.add_scalar("val_acc", val_acc, global_step=epoch)
        print("Epoch {}: testset loss: {}, testset accuracy:{}".format(epoch, val_loss, val_acc))

        if not scheduler is None:
            scheduler.step(val_acc)

    if not scheduler is None:
        if val_acc > best_acc:
            best_acc = val_acc

            if use_gpu:
                torch.save(model.module.state_dict(), os.path.join(log_dir, "%04d.pth" % epoch))
            else:
                torch.save(model.state_dict(), os.path.join(log_dir, "%04d.pth" % epoch))

    return best_acc, (val_loss, val_acc)


def main():
    parser = argparse.ArgumentParser(description="Train a collision detection network.")

    # Dataset
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers.")
    # Optimization
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--gpu_ids", type=str, default='', help="GPUs for running this script.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for gradient descent.")
    parser.add_argument("--factor", type=float, default=0.2, help="Factor by which the learning rate will be reduced.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Number of epochs with no improvement after which learning rate will be reduced.")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Threshold for measuring the new optimum, to only focus on significant changes. ")
    # Misc
    parser.add_argument("--log_dir", type=str, default="../run/", help="Where to save the log?")
    parser.add_argument("--log_name", type=str, required=True, help="Name of the log folder.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--eval_freq", type=int, default=1, help="How frequently to evaluate the model?")

    args = parser.parse_args()

    # Check before run.
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.log_name)

    # Setting up logger
    log_file = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log")
    sys.stdout = Logger(os.path.join(log_dir, log_file))
    print(args)

    for s in args.gpu_ids:
        try:
            int(s)
        except ValueError as e:
            print("Invalid gpu id:{}".format(s))
            raise ValueError

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu_ids)

    if args.gpu_ids:
        if torch.cuda.is_available():
            use_gpu = True
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(args.seed)
        else:
            use_gpu = False
    else:
        use_gpu = False

    torch.manual_seed(args.seed)

    train_datafile = os.path.join(args.data_folder, "train.npy")
    val_datafile = os.path.join(args.data_folder, "test.npy")

    trainloader, trainfullloader, testloader, input_shape = load_dataset(train_datafile, val_datafile, args.batch_size, use_gpu, args.num_workers)
    model = build_model(input_shape)
    train_criterion = BCELoss()
    eval_criterion = BCELoss(reduction="none")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.factor,
                                                           patience=args.patience, verbose=True,
                                                           threshold=args.threshold)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    print("Start training...")
    start = datetime.now()
    best_acc = 0.0

    train_loss_per_eval = []
    val_loss_per_eval = []
    train_acc_per_eval = []
    val_acc_per_eval = []
    with SummaryWriter(log_dir) as writer:
        for epoch in range(args.epochs):
            train(model, trainloader, train_criterion, optimizer, use_gpu, writer, epoch)

            if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
                _, (current_train_loss, current_train_acc) = eval(model, trainfullloader, eval_criterion, None, use_gpu, writer, log_dir, epoch, best_acc)
                best_acc, (current_val_loss, current_val_acc) = eval(model, testloader, eval_criterion, scheduler, use_gpu, writer, log_dir, epoch, best_acc)

                train_loss_per_eval.append(current_train_loss)
                train_acc_per_eval.append(current_train_acc)

                val_loss_per_eval.append(current_val_loss)
                val_acc_per_eval.append(current_val_acc)

    np.save(os.path.join(log_dir, "train_loss_data.npy"), np.array(train_loss_per_eval))
    np.save(os.path.join(log_dir, "train_acc_data.npy"), np.array(train_acc_per_eval))
    np.save(os.path.join(log_dir, "val_loss_data.npy"), np.array(val_loss_per_eval))
    np.save(os.path.join(log_dir, "val_acc_data.npy"), np.array(val_acc_per_eval))

    fig, ax = plt.subplots()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    plt.plot(np.arange(1, len(train_loss_per_eval) + 1), train_loss_per_eval, color="C0", label="train")
    plt.plot(np.arange(1, len(val_loss_per_eval) + 1), val_loss_per_eval, color="C1", label="val")
    plt.legend()

    plt.savefig(os.path.join(log_dir, "loss.pdf"), bbox_inches="tight", dpi=200)
    plt.close("all")

    fig, ax = plt.subplots()
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")

    plt.plot(np.arange(1, len(train_acc_per_eval) + 1), train_acc_per_eval, color="C0", label="train")
    plt.plot(np.arange(1, len(val_acc_per_eval) + 1), val_acc_per_eval, color="C1", label="val")
    plt.legend()

    plt.savefig(os.path.join(log_dir, "accuracy.pdf"), bbox_inches="tight", dpi=200)
    plt.close("all")

    if use_gpu:
        torch.save(model.module.state_dict(), os.path.join(log_dir, "final.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(log_dir, "final.pth"))
    elapsed_time = str(datetime.now() - start)
    print("Finish training. Total elapsed time %s." % elapsed_time)


if __name__ == "__main__":
    main()
