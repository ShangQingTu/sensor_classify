import subprocess
import shutil
import argparse
import os
import re
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='single_mixed',
                        choices=['single', 'single_mixed', 'double_small', 'double_mixed'],
                        help='category of the dataset,single is the panel touched by single finger,single_mixed is data in random order,double_small is the panel touched by 2 fingers with small force amplitude,double_mixed is data in random order')
    parser.add_argument('--model', default='lgb', choices=['lgb', 'svm'],
                        help='classifier to use')
    # max_depth's upper and lower bound
    parser.add_argument('--max_depth_n1', type=int, default=1)
    parser.add_argument('--max_depth_n2', type=int, default=10)

    # svm
    parser.add_argument('--gamma_n1', type=float, default=2)
    parser.add_argument('--gamma_n2', type=float, default=3)
    parser.add_argument('--gamma_interval', type=float, default=0.1)

    parser.add_argument('--C_n1', type=float, default=0.1)
    parser.add_argument('--C_n2', type=float, default=1.1)
    parser.add_argument('--C_interval', type=float, default=0.1)

    args = parser.parse_args()
    run_multi_train(args)


def run_multi_train(args):
    if args.model == "lgb":
        clear_tmp()
        for max_depth in range(args.max_depth_n1, args.max_depth_n2):
            subprocess.run(["python", "train.py", "--category", args.category, "--task", "position", "--model", "lgb",
                            "--max_depth", str(max_depth)])
            print("Done, max_depth: ", max_depth)
        print("lgb position done")
        copy_tmp(args.category, "position", args.model)
        clear_tmp()
        for max_depth in range(args.max_depth_n1, args.max_depth_n2):
            subprocess.run(["python", "train.py", "--category", args.category, "--task", "force", "--model", "lgb",
                            "--max_depth", str(max_depth)])
            print("Done, max_depth: ", max_depth)
        print("lgb force done")
        copy_tmp(args.category, "force", args.model)
    elif args.model == "svm":
        run_multi_svm("position", args)
        run_multi_svm("force", args)


def run_multi_svm(task, args):
    clear_tmp()
    gamma_steps = int((args.gamma_n2 - args.gamma_n1) / args.gamma_interval)
    C_steps = int((args.C_n2 - args.C_n1) / args.C_interval)

    for gamma_step in range(gamma_steps):
        gamma = args.gamma_n1 + args.gamma_interval * gamma_step
        for C_step in range(C_steps):
            C = args.C_n1 + args.C_interval * C_step
            subprocess.run(["python", "train.py", "--category", args.category, "--task", task, "--model", "svm",
                            "--gamma", str(gamma), "--C", str(C)])
    # copy tmp to right path
    copy_tmp(args.category, task, args.model)


def copy_tmp(category, task, model):
    oldname = "tmp.txt"
    newname_txt_path = os.path.join("./data", category, task, model, "milti_train_result.txt")
    newname_csv_path = os.path.join("./data", category, task, model, "milti_train_result.csv")
    shutil.copyfile(oldname, newname_txt_path)

    # turn into csv
    if model == "lgb":
        accuracy_list = []
        max_depth_list = []
        pattern1 = re.compile(r'accuracy: (.*?)\n')
        pattern2 = re.compile(r'max_depth: (.*?),')
        with open(newname_txt_path, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                # traverse Log's every line, use regex to match
                searchObj = re.search(pattern1, line)
                if searchObj:
                    accuracy_list.append(float(searchObj.group(1)))
                searchObj = re.search(pattern2, line)
                if searchObj:
                    max_depth_list.append(int(searchObj.group(1)))

        assert len(accuracy_list) == len(max_depth_list)

        # save csv
        data_dict = {"max_depth": max_depth_list, "accuracy": accuracy_list}
        df = pd.DataFrame(data_dict)
        df.to_csv(newname_csv_path, index=None, encoding='gbk')

    elif model == "svm":
        accuracy_list = []
        gamma_list = []
        C_list = []
        pattern1 = re.compile(r'accuracy: (.*?)\n')
        pattern2 = re.compile(r'gamma: (.*?),')
        pattern3 = re.compile(r'C: (.*?),')
        with open(newname_txt_path, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                # traverse Log's every line, use regex to match
                searchObj = re.search(pattern1, line)
                if searchObj:
                    accuracy_list.append(float(searchObj.group(1)))
                searchObj = re.search(pattern2, line)
                if searchObj:
                    gamma_list.append(float(searchObj.group(1)))
                searchObj = re.search(pattern3, line)
                if searchObj:
                    C_list.append(float(searchObj.group(1)))

        assert len(accuracy_list) == len(gamma_list)
        assert len(accuracy_list) == len(C_list)

        # save csv
        data_dict = {"gamma": gamma_list, "C": C_list, "accuracy": accuracy_list}
        df = pd.DataFrame(data_dict)
        df.to_csv(newname_csv_path, index=None, encoding='gbk')


def clear_tmp():
    # clear tmp.txt
    tmp_fout = open("tmp.txt", "w")
    tmp_fout.close()


if __name__ == '__main__':
    main()
