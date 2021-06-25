import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from test import count_scores
from sklearn.metrics import accuracy_score


def load_csv_data(csv_path, dataset_dir, nolmalize=True):
    # not nolmalize
    raw_trains = pd.read_csv(csv_path)
    if not nolmalize:
        print("not normalized")
        return raw_trains
    nolmalized_dir = os.path.join(dataset_dir, "nomalized")
    # nolmalize
    nolmalized_x_path = os.path.join(nolmalized_dir, "x_train.csv")
    nolmalized_y_path = os.path.join(nolmalized_dir, "y_train.csv")
    nolmalized_trains_x = pd.read_csv(nolmalized_x_path)
    nolmalized_trains_y = pd.read_csv(nolmalized_y_path)
    col_names_x = [column for column in nolmalized_trains_x]
    col_names_x.remove('index')
    col_names_y = [column for column in nolmalized_trains_y]
    col_names_y.remove('index')
    # x drop index
    nolmalized_trains_x_no_index = nolmalized_trains_x.drop("index", axis=1)
    nolmalized_trains_y_no_index = nolmalized_trains_y.drop("index", axis=1)
    nolmalized_trains_xy_no_index = pd.concat([nolmalized_trains_x_no_index, nolmalized_trains_y_no_index], axis=1,
                                              ignore_index=True)
    nolmalized_trains_xy_no_index.columns = col_names_x + col_names_y
    # concat index
    nolmalized_trains_xy = pd.concat([nolmalized_trains_xy_no_index, nolmalized_trains_x['index']], axis=1,
                                     ignore_index=True)
    nolmalized_trains_xy.columns = col_names_x + col_names_y + ['index']
    return nolmalized_trains_xy


def run_lgb(num_class, task_name, dataset_dir, category, nolmalize, max_depth):
    ### path
    csv_path = os.path.join(dataset_dir, "raw.csv")
    lgb_dir = os.path.join(dataset_dir, "lgb")
    if not os.path.exists(lgb_dir):
        os.mkdir(lgb_dir)
    ### load data
    print("load data")
    print("whether normalized: ", nolmalize)
    trains = load_csv_data(csv_path, dataset_dir, nolmalize)
    # The real test set is not this, now the data is labeled, this online is set for no label
    online_test = pd.read_csv(csv_path)
    if (category != "single" and category != "single_mixed") and task_name == "position":
        # position double label
        run_double_rgb(num_class, task_name, trains, online_test, lgb_dir)
    else:
        # single label
        run_single_rgb(num_class, task_name, trains, online_test, lgb_dir, max_depth)


def double_2_single(feature_num=16):
    """
    :param feature_num:
    :return: pos2class : [(pos1,pos2)]
    """
    pos2class = []
    for i in range(feature_num):
        for j in range(i + 1, feature_num):
            pos2class.append((i, j))

    return pos2class


def run_double_rgb(num_class, task_name, trains, online_test, lgb_dir):
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
    run_single_rgb(new_num_class, task_name, single_trains, single_trains, lgb_dir)
    # change single back to double
    res_single_path = os.path.join(lgb_dir, "offline_test.csv")
    csv_data = pd.read_csv(res_single_path, low_memory=False)
    csv_df = pd.DataFrame(csv_data)
    right_data_list = []
    for index, row in csv_df.iterrows():
        row_label_class = row["label"]
        row_pred_class = row["preds"]
        row_label_tuple = pos2class[row_label_class]
        row_pred_tuple = pos2class[row_pred_class]
        right_data_list.append([row_label_tuple[0], row_label_tuple[1], row_pred_tuple[0], row_pred_tuple[1]])

    # gain double label
    right_data_df = pd.DataFrame(right_data_list, columns=label_names + ['pos_1_preds', 'pos_2_preds'])
    double_pos_res = pd.concat([csv_df['index'], right_data_df], axis=1, ignore_index=True)
    double_pos_res.columns = ['index'] + label_names + ['pos_1_preds', 'pos_2_preds']
    final_double_pos_res, final_name = double_pos_res_extend(double_pos_res)
    final_double_pos_res.to_csv(os.path.join(lgb_dir, final_name), index=None, encoding='gbk')


def double_pos_res_extend(double_pos_res):
    """
    :param double_pos_res: (Dataframe) final result
    :return: df add a column: right_num
             name format is val%.3f_0right{}...
    """
    label_names = ['pos_1_label', 'pos_2_label']
    right_count = 0
    right_num_list = []
    for index, row in double_pos_res.iterrows():
        pos_1_preds = min(row["pos_1_preds"], row["pos_2_preds"])
        pos_2_preds = max(row["pos_1_preds"], row["pos_2_preds"])
        pos_1_label = min(row["pos_1_label"], row["pos_2_label"])
        pos_2_label = max(row["pos_1_label"], row["pos_2_label"])
        right_num = 0
        if pos_1_preds == pos_1_label and pos_2_preds == pos_2_label:
            right_num = 2
        elif pos_1_preds == pos_2_label or pos_2_preds == pos_1_label:
            right_num = 1
        right_num_list.append(right_num)
        right_count += right_num
    precision = right_count / (double_pos_res.shape[0] * 2)
    name = 'val%.3f_0right%d_1right%d_2right%d.csv' % (precision, right_num_list.count(0),
                                                       right_num_list.count(1),
                                                       right_num_list.count(2))
    right_data_df = pd.DataFrame(right_num_list, columns=['right_num'])
    df = pd.concat([double_pos_res, right_data_df], axis=1, ignore_index=True)
    df.columns = ['index'] + label_names + ['pos_1_preds', 'pos_2_preds', 'right_num']
    return df, name


def run_single_rgb(num_class, task_name, trains, online_test, lgb_dir, max_depth=4):
    ### split the data, train:valid = 9:1
    print("split the data")
    train_xy, offline_test = train_test_split(trains, test_size=0.1, random_state=21)
    train, val = train_test_split(train_xy, test_size=0.1, random_state=21)
    label_name = "{}_label".format(task_name)
    drop = ['index', label_name]
    print("train set")
    if task_name == "force":
        y = train.force_label  # train set label
        val_y = val.force_label  # valid set label
    elif task_name == "position":
        y = train.position_label
        val_y = val.position_label
    X = train.drop(drop, axis=1)  # train set Characteristic matrix
    print("valid set")
    val_X = val.drop(drop, axis=1)  # valid set Characteristic matrix

    print("test set")
    offline_test_X = offline_test.drop(drop, axis=1)

    ### data process
    lgb_train = lgb.Dataset(X, y, free_raw_data=False)
    lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train, free_raw_data=False)

    ### start training
    seed = 1453
    print('set parameter')
    params = {
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': max_depth,
        'learning_rate': 0.1,
        'random_state': seed,
        'device': 'cpu',
        'objective': 'multiclass',
        'num_class': num_class,
    }
    print("start training")
    gbm = lgb.train(params,  # parameter dict
                    lgb_train,  # train set
                    num_boost_round=2000,  # iterator round
                    valid_sets=lgb_eval,  # valid set
                    early_stopping_rounds=30)  # param of early_stopping
    ### offline predict
    print("offline predict")
    preds_offline_probs = gbm.predict(offline_test_X, num_iteration=gbm.best_iteration)  # output probability
    preds_offline = preds_offline_probs.argmax(axis=1)
    offline = offline_test[['index', label_name]]
    offline = offline.values
    offline = pd.DataFrame(offline, columns=['index', 'label'])

    preds_offline_df = pd.DataFrame(preds_offline, columns=['preds'])
    print("offline.shape", offline.shape)
    print("preds_offline_df.shape", preds_offline_df.shape)
    offline = pd.concat([offline, preds_offline_df], axis=1, ignore_index=True)
    offline.columns = ['index', 'label', 'preds']
    # offline.loc[:,'preds'] = preds_offline_df['prob']
    # offline.label = offline['label'].astype(np.float64)
    offline.label = offline['label']
    offline.to_csv(os.path.join(lgb_dir, "offline_test.csv"), index=None, encoding='gbk')
    score_path = os.path.join(lgb_dir, "scores.txt")
    score_fout = open(score_path, "w")
    count_scores(offline.label, offline.preds, score_fout)
    ### choose character
    print("choose character")
    df = pd.DataFrame(X.columns.tolist(), columns=['feature'])
    df['importance'] = list(gbm.feature_importance())
    df = df.sort_values(by='importance', ascending=False)
    df.to_csv(os.path.join(lgb_dir, "feature_score.csv"), index=None, encoding='gbk')
    # write result
    # accuracy
    tmp_fout = open("tmp.txt", "a")
    tmp_fout.write(
        'max_depth: %d,  accuracy: %.3f\n' % (max_depth, float(accuracy_score(offline.label, offline.preds))))
