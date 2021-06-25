import pandas as pd
import argparse
import os
import xlrd


def normalize_col_test(df, col, mean, std):
    """
    :param df: DataFrame input data
    :param col: column name
    :param mean: mean of the col
    :param std: The variance of this column
    :return:The result of the normalization of this columns
    """
    if std == 0:
        df[col] = df[col].map(lambda x: 0)
    else:
        df[col] = df[col].map(lambda x: (x - mean) / std)
    return df


def get_mean_and_std(df, col):
    mean = df[col].mean()
    std = df[col].std()
    return mean, std


def normalize_col(df, col):
    mean, std = get_mean_and_std(df, col)
    df = normalize_col_test(df, col, mean, std)
    return df


def normalize(data_path, cols):
    """
    :param data_path: csv path
    :param cols:Select columns that need to normalize, and generally remove index and y
    :return:
    """
    print("load data")
    df = pd.read_csv(data_path)

    for col in cols:
        df = normalize_col(df, col)
    return df


def normalize_train(raw_data_path, save_dir, label_names):
    #  normalize The training set
    cols = [str(i) for i in range(1, 1 + 16)]
    normal_df = normalize(raw_data_path, cols)
    x_cols = ['index'] + cols
    df_X = pd.DataFrame(normal_df,
                        columns=x_cols)
    df_y = pd.DataFrame(normal_df, columns=['index'] + label_names)
    df_X.to_csv(os.path.join(save_dir, "x_train.csv"), index=None, sep=',')
    df_y.to_csv(os.path.join(save_dir, "y_train.csv"), index=None, sep=',')


def normalize_test(raw_data_path, feature_num=16):
    # normalize test set
    train_data_path = raw_data_path
    test_data_path = './data/svm_test_set.csv'
    cols = [i for i in range(1, 1 + feature_num)]
    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_path)
    for col in cols:
        mean, std = get_mean_and_std(df_train, col)
        df_test = normalize_col_test(df_test, col, mean, std)
    # result
    df_X = pd.DataFrame(df_test,
                        columns=['index', 'x2', 'x3', 'x4', 'x5', 'x6', 'x12'])
    df_X.to_csv("./data/normalized/x_test.csv", index=None, sep=',')


def creat_dataset(args, force_map={"double_small": 0, "double_middle": 1, "double_big": 2}):
    # read excel
    raw_data = xlrd.open_workbook(args.raw_data_path)
    # select sheet1
    table = raw_data.sheet_by_index(0)

    print("total rows:" + str(table.nrows))
    print("total cols:" + str(table.ncols))
    print("first row:" + str(table.row_values(0)))
    # quit()
    if args.category == 'single':
        # how many force num in data
        one_force_data_num = args.repeat_num * args.feature_num
        force_data_list = []
        position_data_list = []

        for force_split in range(args.force_num):

            for row_num in range(one_force_data_num * force_split, one_force_data_num * (force_split + 1)):
                # ignore first row
                sample_row_num = row_num + force_split + 1
                # select 16 features
                row_data = table.row_values(sample_row_num)
                row_features = row_data[1:1 + args.feature_num]

                force_data = row_features + [force_split, row_num]
                force_data_list.append(force_data)
                position_data = row_features + [row_data[0] - 1, row_num]
                position_data_list.append(position_data)
        # save
        col_feature_names = table.row_values(0)[1:1 + args.feature_num]
        col_feature_names = [str(int(name)) for name in col_feature_names]
        force_data_df = pd.DataFrame(force_data_list, columns=col_feature_names + ['force_label', 'index'])
        position_data_df = pd.DataFrame(position_data_list, columns=col_feature_names + ['position_label', 'index'])
        force_raw_data_path = os.path.join(args.force_set_dir, "raw.csv")
        position_raw_data_path = os.path.join(args.position_set_dir, "raw.csv")
    elif args.category == 'double_small':
        force_data_list = []
        position_data_list = []
        for row_num in range(table.nrows - 1):
            # ignore first row
            row_data = table.row_values(row_num + 1)
            # top2 is pos_x, pos_y
            row_features = row_data[2:2 + args.feature_num]
            force_num = force_map[args.category]
            force_data = row_features + [force_num, row_num]
            force_data_list.append(force_data)
            position_data = row_features + [row_data[0] - 1, row_data[1] - 1, row_num]
            position_data_list.append(position_data)

        col_feature_names = table.row_values(0)[2:2 + args.feature_num]
        col_feature_names = [str(int(name)) for name in col_feature_names]
        force_data_df = pd.DataFrame(force_data_list, columns=col_feature_names + ['force_label', 'index'])
        position_data_df = pd.DataFrame(position_data_list,
                                        columns=col_feature_names + ['pos_1_label', 'pos_2_label', 'index'])
        force_raw_data_path = os.path.join(args.force_set_dir, "raw.csv")
        position_raw_data_path = os.path.join(args.position_set_dir, "raw.csv")
    elif args.category == 'single_mixed':

        force_data_list = []
        position_data_list = []

        for row_num in range(table.nrows):
            # Just read the data from the first line
            # Sixteen features were taken out
            row_data = table.row_values(row_num)
            row_features = row_data[1:1 + args.feature_num]

            # Because it is a mixed dataset, the first column is the location, and the column after Feature is Force
            force_data = row_features + [row_data[1 + args.feature_num], row_num]
            force_data_list.append(force_data)
            position_data = row_features + [row_data[0] - 1, row_num]
            position_data_list.append(position_data)
        # save
        col_feature_names = [i for i in range(1, 1 + args.feature_num)]
        # change column name from int to str
        col_feature_names = [str(int(name)) for name in col_feature_names]
        force_data_df = pd.DataFrame(force_data_list, columns=col_feature_names + ['force_label', 'index'])
        position_data_df = pd.DataFrame(position_data_list, columns=col_feature_names + ['position_label', 'index'])
        force_raw_data_path = os.path.join(args.force_set_dir, "raw.csv")
        position_raw_data_path = os.path.join(args.position_set_dir, "raw.csv")

    elif args.category == 'double_mixed':
        force_data_list = []
        position_data_list = []
        for row_num in range(table.nrows):
            row_data = table.row_values(row_num)
            # top2 is pos_x, pos_y
            row_features = row_data[2:2 + args.feature_num]
            force_num = row_data[2 + args.feature_num]
            force_data = row_features + [force_num, row_num]
            force_data_list.append(force_data)
            position_data = row_features + [row_data[0] - 1, row_data[1] - 1, row_num]
            position_data_list.append(position_data)

        # save
        col_feature_names = [i for i in range(1, 1 + args.feature_num)]
        # change column name from int to str
        col_feature_names = [str(int(name)) for name in col_feature_names]
        force_data_df = pd.DataFrame(force_data_list, columns=col_feature_names + ['force_label', 'index'])
        position_data_df = pd.DataFrame(position_data_list,
                                        columns=col_feature_names + ['pos_1_label', 'pos_2_label', 'index'])
        force_raw_data_path = os.path.join(args.force_set_dir, "raw.csv")
        position_raw_data_path = os.path.join(args.position_set_dir, "raw.csv")

    force_data_df.to_csv(force_raw_data_path, index=None, sep=',')
    position_data_df.to_csv(position_raw_data_path, index=None, sep=',')


def work(args):
    # create dataset
    creat_dataset(args)
    # Normalize the SVM data
    force_raw_data_path = os.path.join(args.force_set_dir, "raw.csv")
    normalize_train(force_raw_data_path, args.nomalized_force_set_dir, ['force_label'])
    position_raw_data_path = os.path.join(args.position_set_dir, "raw.csv")
    if args.category == 'single' or args.category == 'single_mixed':
        normalize_train(position_raw_data_path, args.nomalized_position_set_dir, ['position_label'])
    else:
        # 2 key
        normalize_train(position_raw_data_path, args.nomalized_position_set_dir, ['pos_1_label', 'pos_2_label'])
    print("Done")


def parse_args():
    parser = argparse.ArgumentParser()
    # path parameters
    parser.add_argument('--category', default='double_mixed',
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
    parser.add_argument('--mixed', action='store_true',
                        help='whether to mix the random order data')

    args = parser.parse_args()

    # The location of the original data set
    raw_data_file_name = "{}_set.xlsx".format(args.category)
    args.raw_data_path = os.path.join(args.data_dir, raw_data_file_name)
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
    # dir preparation
    args = parse_args()
    work(args)
