from sklearn.model_selection import train_test_split
import pandas as pd
import os
from test import count_scores
from lgb_model import load_csv_data, double_2_single, double_pos_res_extend
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def run_svm(num_class, task_name, dataset_dir, category, nolmalize, gamma, C):
    ### path
    csv_path = os.path.join(dataset_dir, "raw.csv")
    svm_dir = os.path.join(dataset_dir, "svm")
    if not os.path.exists(svm_dir):
        os.mkdir(svm_dir)
    ### load data
    print("load data")
    print("whether normalized: ", nolmalize)
    trains = load_csv_data(csv_path, dataset_dir, nolmalize)
    online_test = pd.read_csv(csv_path)
    if (category != "single" and category != "single_mixed") and task_name == "position":
        run_double_svm(num_class, task_name, trains, online_test, svm_dir)
    else:
        run_single_svm(num_class, task_name, trains, online_test, svm_dir, gamma, C)


def run_double_svm(num_class, task_name, trains, online_test, svm_dir):
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
    run_single_svm(new_num_class, task_name, single_trains, single_trains, svm_dir)
    # change single back to double
    res_single_path = os.path.join(svm_dir, "offline_test.csv")
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
    final_double_pos_res.to_csv(os.path.join(svm_dir, final_name), index=None, encoding='gbk')


def run_single_svm(num_class, task_name, trains, online_test, svm_dir, gamma=5, C=1):
    ### split the data, train:valid = 9:1
    print("split the data")
    train_xy, offline_test = train_test_split(trains, test_size=0.1, random_state=21)

    train = train_xy
    label_name = "{}_label".format(task_name)
    drop = ['index', label_name]
    print("train set")
    if task_name == "force":
        y = train.force_label
    elif task_name == "position":
        y = train.position_label
    X = train.drop(drop, axis=1)
    print("test set")
    offline_test_X = offline_test.drop(drop, axis=1)



    seed = 1453
    print('set parameter')
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
    ])
    print("start training")
    rbf_kernel_svm_clf.fit(X, y)

    ### offline predict
    print("offline predict")
    preds_offline = rbf_kernel_svm_clf.predict(offline_test_X)
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
    offline.to_csv(os.path.join(svm_dir, "offline_test.csv"), index=None, encoding='gbk')
    score_path = os.path.join(svm_dir, "scores.txt")
    score_fout = open(score_path, "w")
    count_scores(offline.label, offline.preds, score_fout)
    # accuracy
    tmp_fout = open("tmp.txt", "a")
    tmp_fout.write(
        'gamma: %.3f, C: %.3f,  accuracy: %.3f\n' % (gamma, C, float(accuracy_score(offline.label, offline.preds))))
