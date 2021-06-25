import torch
from collections import defaultdict, OrderedDict
from collections import deque
import sys
import argparse
import os
import logging
import re
import matplotlib.pyplot as plt
import pandas as pd


def setup_logger(name, save_dir, filename="log.txt"):
    """
    :param name: name of log
    :param save_dir: directory to save log
    :param filename: log file name
    :return: looger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger, fh


def strip_prefix_if_present(state_dict, prefix):
    """
    only used in test.py
    :param state_dict: loaded ckpt's state dictionary
    :param prefix:
    :return:a state dictionary with the key removed from the prefix name
    """
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    """
     Converts the data stored during training to STR classes
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


def draw_double_picture(log_path, save_dir, pic_type):
    """
    :param log_path: log.txt's path
    :param save_dir: directory to save log
    :param filename: log file name
    :return:
    """
    pattern1 = re.compile(r'progress: (.*?)  loss')
    pattern2 = re.compile(r'loss: (.*?) \(')
    pattern3 = re.compile(r'precision: ')
    pattern_acc = re.compile(r'accuracy: (.*?)\n')
    progress_list = []
    loss_list = []
    every_accuracy_list = []
    accuracy_list = []
    with open(log_path, "r") as fin:
        lines = fin.readlines()
        print("pic_type: ", pic_type)
        print("len(lines): ", len(lines))
        for line in lines:
            searchObj = re.search(pattern1, line)
            if searchObj:
                progress_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern2, line)
            if searchObj:
                loss_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern_acc, line)
            if searchObj:
                every_accuracy_list.append(float(searchObj.group(1)))
            # searchObj = re.search(pattern3, line)
            # if searchObj:
            #     end_pos = searchObj.span()[1]
            #     no_enter_accuracy = line[end_pos:len(line) - 1]
            #     accuracy_list.append(float(no_enter_accuracy))
        assert len(loss_list) == len(progress_list)
        assert len(loss_list) == len(every_accuracy_list)

    print("len(every_accuracy_list): ", len(every_accuracy_list))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # set title of picture as pic_type
    fig.suptitle(pic_type, fontsize=14, fontweight='bold')
    # x name
    ax.set_xlabel("epoch")
    if pic_type == 'loss':
        # y name
        ax.set_ylabel("loss")
        ax.plot(progress_list, loss_list, color='blue')
    elif pic_type == 'accuracy_every_epoch':
        ax.set_ylabel("accuracy")
        epoch_list = [i for i in range(len(accuracy_list))]
        ax.plot(epoch_list, accuracy_list, color='blue')
    elif pic_type == 'accuracy':
        ax.set_ylabel("accuracy")
        ax.plot(progress_list, every_accuracy_list, color='blue')
    elif pic_type == 'accuracy and loss':
        ax.set_ylabel("accuracy;loss")
        ax.plot(progress_list, every_accuracy_list, color='blue', label="accuracy")
        ax.plot(progress_list, loss_list, color='green', label="loss")

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig(os.path.join(save_dir, "{}.png".format(pic_type)))

    # plt.show()
    # quit()

    # save csv
    data_dict = {"progress": progress_list, "loss": loss_list, "accuracy": every_accuracy_list}
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(save_dir, "loss_and_accuracy.csv"), index=None, encoding='gbk')


def draw_single_picture(log_path, save_dir, pic_type):
    """
    :param log_path: log.txt's path
    :param save_dir: directory to save picture
    :param pic_type: picture type
    :return:
    """
    pattern1 = re.compile(r'progress: (.*?)  train_loss')
    pattern2 = re.compile(r'train_loss: (.*?)  train_accuracy')
    pattern3 = re.compile(r'train_accuracy: (.*?)  valid_accuracy')
    pattern4 = re.compile(r'valid_accuracy: (.*?)  valid_loss')
    pattern5 = re.compile(r'valid_loss: (.*?)\n')
    progress_list = []
    train_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []
    valid_loss_list = []

    with open(log_path, "r") as fin:
        lines = fin.readlines()
        print("pic_type: ", pic_type)
        print("len(lines): ", len(lines))
        for line in lines:
            searchObj = re.search(pattern1, line)
            if searchObj:
                progress_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern2, line)
            if searchObj:
                train_loss_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern3, line)
            if searchObj:
                train_accuracy_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern4, line)
            if searchObj:
                valid_accuracy_list.append(float(searchObj.group(1)))
            searchObj = re.search(pattern5, line)
            if searchObj:
                valid_loss_list.append(float(searchObj.group(1)))

    print("len(valid_loss_list): ", len(valid_loss_list))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle(pic_type, fontsize=14, fontweight='bold')
    ax.set_xlabel("epoch")

    if pic_type == 'accuracy and loss':
        ax.set_ylabel("accuracy;loss")
        ax.plot(progress_list, train_loss_list, color='blue', label=" train_loss")
        ax.plot(progress_list, train_accuracy_list, color='green', label="train_accuracy")
        ax.plot(progress_list, valid_accuracy_list, color='red', label="valid_accuracy")
        ax.plot(progress_list, valid_loss_list, color='pink', label="valid_loss")

    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.savefig(os.path.join(save_dir, "{}.png".format(pic_type)))

    # plt.show()
    # quit()

    data_dict = {"progress": progress_list,
                 "train_loss": train_loss_list,
                 "train_accuracy": train_accuracy_list,
                 "valid_accuracy": valid_accuracy_list,
                 "valid_loss": valid_loss_list
                 }
    df = pd.DataFrame(data_dict)
    df.to_csv(os.path.join(save_dir, "loss_and_accuracy.csv"), index=None, encoding='gbk')


def work(args):
    # name for picture
    name1 = args.log_path.replace("/", "_")
    name2 = name1.replace("txt", "")
    name3 = name2.replace(".", "")
    pic_save_dir = os.path.join(args.save_dir, name3)
    if not os.path.exists(pic_save_dir):
        os.mkdir(pic_save_dir)

    if args.category != "single_mixed":
        if args.pic_type == "both":
            # loss and accuracy will be ploted in one picture
            draw_double_picture(args.log_path, pic_save_dir, 'accuracy and loss')
        else:
            draw_double_picture(args.log_path, pic_save_dir, args.pic_type)
    else:
        if args.pic_type == "both":
            # loss and accuracy will be ploted in one picture
            draw_single_picture(args.log_path, pic_save_dir, 'accuracy and loss')
        else:
            draw_single_picture(args.log_path, pic_save_dir, args.pic_type)

    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--log_path', required=True,
                        help='log file path')
    parser.add_argument('--pic_type', default='both', choices=['loss', 'accuracy', 'both'],
                        help='picture type to draw')

    parser.add_argument('--category', default='single_mixed',
                        choices=['single', 'single_mixed', 'double_small', 'double_mixed'],
                        help='category of the dataset,single is the panel touched by single finger,single_mixed is data in random order,double_small is the panel touched by 2 fingers with small force amplitude,double_mixed is data in random order')


    parser.add_argument('--save_dir', default='./figure/',
                        help='picture directory to save')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    return args


if __name__ == '__main__':
    # dir preparation
    args = parse_args()
    work(args)
