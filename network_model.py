# coding=UTF-8
from sklearn.model_selection import train_test_split
from lgb_model import load_csv_data, double_2_single
import torch
import torch.utils.data as Data
from torch import nn, optim
import random
import numpy as np
import shutil
import os
import pandas as pd
from model import ClassifierModel
from utils import setup_logger, MetricLogger, strip_prefix_if_present


def run_network(num_class, task_name, dataset_dir, category, nolmalize, args):
    ### path
    csv_path = os.path.join(dataset_dir, "raw.csv")
    network_dir = os.path.join(dataset_dir, "network")
    if not os.path.exists(network_dir):
        os.mkdir(network_dir)
    ### load data
    print("load data")
    print("whether normalized: ", nolmalize)
    trains = load_csv_data(csv_path, dataset_dir, nolmalize)
    # The real test set is not this, now the data is labeled, this online is set for no label
    online_test = pd.read_csv(csv_path)
    if (category != "single" and category != "single_mixed") and task_name == "position":
        # position double label
        run_double_network(num_class, task_name, trains, online_test, network_dir, args)
    else:
        # single label
        run_single_network(num_class, task_name, trains, online_test, network_dir, args)


def run_double_network(num_class, task_name, trains, online_test, network_dir, args):
    ### split the data, train:valid = 9:1
    print("split the data")
    pos2class = double_2_single()
    new_class_list = []
    for index, row in trains.iterrows():
        new_tuple = (row["pos_1_label"], row["pos_2_label"])
        new_class = pos2class.index((new_tuple))
        new_class_list.append(new_class)
    label_names = ['pos_1_label', 'pos_2_label']
    # drop old label
    no_double_trains = trains.drop(label_names, axis=1)
    # gain new label
    new_class_df = pd.DataFrame(new_class_list, columns=['label'])
    single_trains = pd.concat([no_double_trains, new_class_df], axis=1, ignore_index=True)
    label_name = "{}_label".format(task_name)
    single_trains.columns = [str(num) for num in range(16)] + ['index', label_name]
    # same like single label classify
    new_num_class = len(pos2class)
    if args.not_use_top2:
        run_single_network(new_num_class, task_name, single_trains, single_trains, network_dir, args)
    else:
        # use top 2
        run_single_network(num_class, task_name, trains, trains, network_dir, args)


def run_single_network(num_class, task_name, trains, online_test, network_dir, args):
    ### split the data, train:valid = 9:1
    print("split the data")
    train, val = train_test_split(trains, test_size=0.1, random_state=21)
    if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or task_name == "force":
        label_name = "{}_label".format(task_name)
        drop = ['index', label_name]
        print("train set")
        if task_name == "force":
            y = train.force_label
            val_y = val.force_label
        elif task_name == "position":
            y = train.position_label
            val_y = val.position_label
        X = train.drop(drop, axis=1)
        print("valid set")
        val_X = val.drop(drop, axis=1)

        print("test set")
        args.num_class = num_class
        # df to tensor
        train_x = torch.Tensor(np.array(X))
        train_y = torch.Tensor(np.array(y))
        val_x_tensor = torch.Tensor(np.array(val_X))
        val_y_tensor = torch.Tensor(np.array(val_y))
        work(args, train_x, train_y, val_x_tensor, val_y_tensor, val[['index', label_name]], network_dir)
    else:
        drop = ['index', 'pos_1_label', 'pos_2_label']
        print("train set")
        y1 = train.pos_1_label
        y2 = train.pos_2_label
        val_y1 = val.pos_1_label
        val_y2 = val.pos_2_label
        X = train.drop(drop, axis=1)
        print("valid set")
        val_X = val.drop(drop, axis=1)
        print("test set")
        # set num_class, *2 in order to let the sieze of ANN's output change from num_class to 2×num_class,
        # we will get 2 tonser of num_class size,which represents x and y
        args.num_class = num_class * 2
        train_x = torch.Tensor(np.array(X))
        train_y1 = torch.Tensor(np.array(y1))
        train_y2 = torch.Tensor(np.array(y2))
        val_x_tensor = torch.Tensor(np.array(val_X))
        val_y1_tensor = torch.Tensor(np.array(val_y1))
        val_y2_tensor = torch.Tensor(np.array(val_y2))
        work(args, train_x, (train_y1, train_y2), val_x_tensor, (val_y1_tensor, val_y2_tensor),
             val[['index', 'pos_1_label', 'pos_2_label']], network_dir)


def train(inputs, outputs, val_x_tensor, val_y_tensor, val_df, args, logger, network_dir):
    """

    :param inputs:(tensor) as input tensor
    :param outputs:(tensor) as labeled tensor
    :param val_x_tensor: valid input tensor
    :param val_y_tensor: valid labeled tensor
    :param val_df: dataframe of valid set's index and label
    :param args:some parameters
    :param logger: save at network_dir/log.txt
    :param network_dir: directory to save log
    :return:end of training
    """

    logger.info('[1] Building model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set model
    model = ClassifierModel(num_class=args.num_class,
                            dropout=args.dropout,
                            dim_h=args.dim_h).to(device)

    logger.info(model)
    # optimizer
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    meters = MetricLogger(delimiter="  ")
    # loss function
    criterion = nn.CrossEntropyLoss()
    # scheduler will make the learning rate *0.1 when epoch is schedule_step. Now only do this at epoch 3
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.schedule_step], gamma=0.1)
    if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or args.task == "force":
        torch_dataset = Data.TensorDataset(inputs, outputs)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)
        logger.info('[2] Start training......')
        for epoch_num in range(args.max_epoch):
            # example_num:a epoch need how many batch
            example_num = outputs.shape[0] // args.batch_size
            for batch_iter, (input_tensor, label) in enumerate(loader):
                progress = epoch_num + batch_iter / example_num
                optimizer.zero_grad()
                pred = model(input_tensor.to(device).view(args.batch_size, -1))
                # set label
                if label.shape[0] != args.batch_size:
                    logger.info('last dummy batch')
                    break
                label = label.view(args.batch_size)
                label = label.to(device)
                loss = criterion(pred, label.long())

                loss.backward()
                optimizer.step()
                meters.update(loss=loss)
                if (batch_iter + 1) % (example_num // 10) == 0:
                    precision0, pred_list0, right_num_list0 = validate(model, device, args, val_x_tensor, val_y_tensor)
                    precision1, pred_list1, right_num_list1 = validate(model, device, args, inputs, outputs)
                    train_loss = get_valid_loss(model, device, args, inputs, outputs)
                    valid_loss = get_valid_loss(model, device, args, val_x_tensor, val_y_tensor)
                    logger.info(
                        meters.delimiter.join(
                            [
                                "progress: {prog:.2f}",
                                "train_loss: {train_loss:.4f}",
                                "train_accuracy: {train_accuracy:.2f}",
                                "valid_accuracy: {valid_accuracy:.2f}",
                                "valid_loss: {valid_loss:.4f}"
                            ]
                        ).format(
                            prog=progress,
                            train_loss=train_loss,
                            train_accuracy=precision1,
                            valid_accuracy=precision0,
                            valid_loss=valid_loss
                        )
                    )

            precision, pred_list, right_num_list = validate(model, device, args, val_x_tensor, val_y_tensor)
            logger.info("val")
            logger.info("precision: {}".format(precision.numpy()))

            scheduler.step()

            save_valid_result(pred_list, val_df,
                              os.path.join(network_dir, 'model_epoch%d_val%.3f.csv' % (epoch_num, precision)))
    else:
        torch_dataset = Data.TensorDataset(inputs, outputs[0], outputs[1])
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True)

        logger.info('[2] Start training......')
        for epoch_num in range(args.max_epoch):
            # example_num:a epoch need how many batch
            example_num = outputs[0].shape[0] // args.batch_size
            for batch_iter, (input_tensor, label1, label2) in enumerate(loader):
                progress = epoch_num + batch_iter / example_num
                optimizer.zero_grad()
                # forward
                pred_raw = model(input_tensor.to(device))
                pred = pred_raw.view(args.batch_size * 2, -1)
                # set label
                if label1.shape[0] != args.batch_size:
                    logger.info('last dummy batch')
                    break
                label1 = label1.view(args.batch_size)
                label2 = label2.view(args.batch_size)
                label = torch.cat((label1, label2)).to(device)
                loss = criterion(pred, label.long())

                loss.backward()
                optimizer.step()
                meters.update(loss=loss)
                if (batch_iter + 1) % (example_num // 100) == 0:
                    precision0, pred_list0, right_num_list0 = validate(model, device, args, val_x_tensor, val_y_tensor)
                    logger.info(
                        meters.delimiter.join(
                            [
                                "progress: {prog:.2f}",
                                "{meters}", "accuracy: {accuracy:.2f}"
                            ]
                        ).format(
                            prog=progress,
                            meters=str(meters),
                            accuracy=precision0
                        )
                    )

            precision, pred_list, right_num_list = validate(model, device, args, val_x_tensor, val_y_tensor)
            logger.info("val")
            logger.info("precision: {}".format(precision))

            scheduler.step()

            # save csv
            save_valid_result_double(pred_list, val_df, right_num_list,
                                     os.path.join(network_dir, 'model_epoch%d_val%.3f_0right%d_1right%d_2right%d.csv'
                                                  % (epoch_num, precision, right_num_list.count(0),
                                                     right_num_list.count(1),
                                                     right_num_list.count(2))))


def save_valid_result(pred_list, index_df, save_path):
    index_df = index_df.values
    index_df = pd.DataFrame(index_df, columns=['index', 'label'])
    preds_offline_df = pd.DataFrame(pred_list, columns=['preds'])
    offline = pd.concat([index_df, preds_offline_df], axis=1, ignore_index=True)
    offline.columns = ['index', 'label', 'preds']
    offline.to_csv(os.path.join(save_path), index=None, encoding='gbk')


def save_valid_result_double(pred_list, index_df, right_num_list, save_path):
    index_df = index_df.values
    index_df = pd.DataFrame(index_df, columns=['index', 'label1', 'label2'])
    preds_offline_df = pd.DataFrame(pred_list, columns=['pred1', 'pred2'])
    right_df = pd.DataFrame(right_num_list, columns=['right_num'])
    offline = pd.concat([index_df, preds_offline_df, right_df], axis=1, ignore_index=True)
    offline.columns = ['index', 'label1', 'label2', 'pred1', 'pred2', 'right_num']
    offline.to_csv(os.path.join(save_path), index=None, encoding='gbk')


def get_valid_loss(model, device, args, inputs, outputs):
    """
    :param
    - inputs: (tensor) input tensor
    - outputs: (tensor) labeled tensor
    - args: some parameters
    :return:
    -   loss  on valid set
    """
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    data_num = outputs.shape[
        0] if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or args.task == "force" else \
        outputs[0].shape[0]

    if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or args.task == "force":
        torch_dataset = Data.TensorDataset(inputs, outputs)
        # no shuffle
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
        pred_prob_list = []
        for batch_iter, (input_tensor, label) in enumerate(loader):
            pred_prob = model(input_tensor.to(device).view(-1))
            pred_prob_list.append(pred_prob)

        pred_prob_tensor = torch.stack(pred_prob_list).to(device)
        loss = criterion(pred_prob_tensor, outputs.long()) / data_num

        return loss


def validate(model, device, args, inputs, outputs):
    """
    :param
    - inputs: (tensor) input tensor
    - outputs: (tensor) labeled tensor
    - args: some parameters
    :return:
    - accuracy and predicted class for every data
    """
    batch_size = args.batch_size
    data_num = outputs.shape[
        0] if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or args.task == "force" else \
        outputs[0].shape[0]
    pred_list = []
    right_num_list = []
    with torch.no_grad():
        if args.not_use_top2 or args.category == 'single_mixed' or args.category == 'single' or args.task == "force":
            torch_dataset = Data.TensorDataset(inputs, outputs)
            # no shuffle
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
            # right predict number
            right_count = 0
            for batch_iter, (input_tensor, label) in enumerate(loader):
                pred_prob = model(input_tensor.to(device).view(batch_size, -1))
                pred_class = torch.argmax(pred_prob, 1)
                pred_list.append(pred_class.cpu().numpy())
                right_count += (pred_class.cpu() == label).sum()
        else:
            torch_dataset = Data.TensorDataset(inputs, outputs[0], outputs[1])
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=False)
            # right predict number
            right_count = 0
            for batch_iter, (input_tensor, label1, label2) in enumerate(loader):
                pred_prob = model(input_tensor.to(device).view(batch_size, -1))
                pred = pred_prob.view(args.batch_size * 2, -1)
                pred_classes = torch.argmax(pred, 1).cpu()
                label = torch.cat((label1, label2))
                if args.not_use_single_precision:
                    # both right is right for double
                    if pred_classes[0] == label[0] and pred_classes[1] == label[1]:
                        right_count += 1
                else:
                    # 1 right is right for single
                    single_right_num = (pred_classes == label).sum().item()
                    reverse_np = pred_classes.numpy()[::-1]
                    label_np = label.numpy()
                    reverse_single_right_num = np.sum(reverse_np == label_np)
                    # set right_num
                    right_num = max(single_right_num, reverse_single_right_num)
                    right_num_list.append(right_num)
                    right_count += right_num
                pred_list.append(pred_classes.numpy())
        if not args.not_use_single_precision and not args.not_use_top2 and args.task != "force" \
                and args.category != 'single_mixed' \
                and args.category != 'single':
            # Because every piece of data has two keys to predict, the denominator is going to be twice the amount of data
            precision = right_count / float(data_num * 2)
        else:
            precision = right_count / float(data_num)

    return precision, pred_list, right_num_list


def work(args, train_inputs, train_outputs, val_x, val_y, val_df, network_dir):
    """
    :param args: some parameters
    :return: end of training
    """
    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.not_use_top2:
        network_dir = os.path.join(network_dir, "not_use_top2")

    if os.path.isdir(network_dir):
        shutil.rmtree(network_dir)
    os.makedirs(network_dir)

    logger, fh = setup_logger("Classify", network_dir)

    # args display
    for k, v in vars(args).items():
        logger.info(k + ':' + str(v))

    train(train_inputs, train_outputs, val_x, val_y, val_df, args, logger, network_dir)
    # close logger's filerHandler, prevent log from being covered
    logger.removeHandler(fh)
