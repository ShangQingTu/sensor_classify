import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def count_scores(groundtruth_y_np, pridict_y_np, log_fout):
    # precision
    log_fout.write("micro_precision: %.3f , macro_precision: %.3f  \n" % (
        precision_score(groundtruth_y_np, pridict_y_np, average='micro'),
        precision_score(groundtruth_y_np, pridict_y_np, average='macro')))
    log_fout.write("pricision for all classes:\n")
    log_fout.write(str(precision_score(groundtruth_y_np, pridict_y_np, average=None)))
    # recall
    log_fout.write("\n micro_recall: %.3f , macro_recall: %.3f  \n" % (
        float(recall_score(groundtruth_y_np, pridict_y_np, average='micro')),
        float(recall_score(groundtruth_y_np, pridict_y_np, average='macro'))))
    log_fout.write("recall for all classes:\n")
    log_fout.write(str(recall_score(groundtruth_y_np, pridict_y_np, average=None)))
    # f1
    log_fout.write("\n micro_f1: %.3f , macro_f1: %.3f  \n" % (
        float(f1_score(groundtruth_y_np, pridict_y_np, average='micro')),
        float(f1_score(groundtruth_y_np, pridict_y_np, average='macro'))))
    log_fout.write("f1 for all classes:\n")
    log_fout.write(str(f1_score(groundtruth_y_np, pridict_y_np, average=None)))
    # accuracy
    log_fout.write('\n Valid Result:  Accuracy: %.3f\n' % (float(accuracy_score(groundtruth_y_np, pridict_y_np))))


def loadModel(filePath):
    """
    :param filePath: pkl path
    :return: model loaded
    """
    fin = open(filePath, "rb")
    return pickle.load(fin)


