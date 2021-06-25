from lgb_model import run_lgb
from svm_model import run_svm
from network_model import run_network
import os
import argparse


def work_lgb(args):
    # lgb is a kind of decision tree
    if args.task == 'both':
        run_lgb(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args.max_depth)
        run_lgb(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args.max_depth)
    elif args.task == 'force':
        run_lgb(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args.max_depth)
    elif args.task == 'position':
        run_lgb(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args.max_depth)


def work_svm(args):
    # training
    # svm is support vector machine`
    if args.task == 'both':
        run_svm(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args.gamma, args.C)
        run_svm(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args.gamma, args.C)
    elif args.task == 'force':
        run_svm(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args.gamma, args.C)
    elif args.task == 'position':
        run_svm(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args.gamma, args.C)


def work_network(args):
    # a kind of neural network
    if args.task == 'both':
        run_network(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args)
        run_network(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args)
    elif args.task == 'force':
        run_network(args.force_num, "force", args.force_set_dir, args.category, args.normalize, args)
    elif args.task == 'position':
        run_network(args.position_num, "position", args.position_set_dir, args.category, args.normalize, args)


def work(args):
    # create dataset
    if args.model == "lgb":
        work_lgb(args)
    elif args.model == "svm":
        work_svm(args)
    elif args.model == "network":
        #  neural network
        work_network(args)
    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='single_mixed',
                        choices=['single', 'single_mixed', 'double_small', 'double_mixed'],
                        help='category of the dataset,single is the panel touched by single finger,single_mixed is data in random order,double_small is the panel touched by 2 fingers with small force amplitude,double_mixed is data in random order')
    parser.add_argument('--task', default='both', choices=['force', 'position', 'both'],
                        help='task to predict')
    parser.add_argument('--data_dir', default='./data/',
                        help='directory of dataset')

    # dataset parameters
    parser.add_argument('--feature_num', type=int, default=16,
                        help='feature num')
    parser.add_argument('--force_num', type=int, default=3,
                        help='force num')

    parser.add_argument('--position_num', type=int, default=16,
                        help='position num')

    parser.add_argument('--repeat_num', type=int, default=3,
                        help='repeat label num')

    # model parameters
    parser.add_argument('--not_normalize', action='store_true',
                        help='whether to normalize, defult is yes')

    parser.add_argument('--model', default='network', choices=['lgb', 'svm', 'network'],
                        help='classifier to use')

    # lgb  parameters
    parser.add_argument('--max_depth', type=int, default=4)

    # svm parameters
    parser.add_argument('--gamma', type=float, default=5)
    parser.add_argument('--C', type=float, default=1)

    # model parameters, see them in `model.py`
    parser.add_argument('--not_use_top2', action='store_true',
                        help='hether to use 2 prob vectors as the basis of double bond prediction')
    parser.add_argument('--not_use_single_precision', action='store_true',
                        help='Whether to use each single bond statistic as the precision of the double bond prediction')
    parser.add_argument('--dim_h', type=int, default=1024)

    # training parameters, These are the parameters for neural network training
    parser.add_argument('--max_epoch', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--schedule_step', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=3e-6, help='weight decay rate per batch')
    parser.add_argument('--seed', type=int, default=666666, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--optim', default='adam', choices=['adam', 'adamw', 'sgd'])

    args = parser.parse_args()

    args.normalize = not bool(args.not_normalize)
    # The location of the processed data set
    args.category_data_dir = os.path.join(args.data_dir, args.category)
    if not os.path.exists(args.category_data_dir):
        os.mkdir(args.category_data_dir)

    # The path where the generated force data set is placed
    args.force_set_dir = os.path.join(args.category_data_dir, "force")
    # The path where the generated location data set is placed
    args.position_set_dir = os.path.join(args.category_data_dir, "position")
    # The position of the data set after normalization
    args.nomalized_force_set_dir = os.path.join(args.force_set_dir, "nomalized")
    args.nomalized_position_set_dir = os.path.join(args.position_set_dir, "nomalized")
    for new_path in [args.force_set_dir, args.position_set_dir, args.nomalized_force_set_dir,
                     args.nomalized_position_set_dir]:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    return args


if __name__ == '__main__':
    args = parse_args()
    work(args)
