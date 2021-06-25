import torch
import torch.nn as nn
from torch.nn.functional import softmax


class ClassifierModel(nn.Module):
    """

    ClassifierModel: 这里的结构就是参照2篇传感器论文里常见的神经网络结果写的
    特征向量 先
    输入到 第一个全连接层fc_layer0, relu激活一次并dropout一次,向量维度变为1024
    再输入到 第二个全连接层fc_layer1,输出维度为512
    ...
    最后输出一个 self.nclass 的向量,就是分类的每个类对应的概率
    """

    def __init__(self, num_class, dropout, dim_h=1024):
        """
        :arg:
        num_class 分类的类数(二分类比较特殊,不需要用独热向量,所以就输出1维就行)
        dropout  对输入的一些维度进行随机遗忘的比例,为了防止过拟合
        dim_h 第一个隐藏层的维度
        """
        super(ClassifierModel, self).__init__()
        self.nclass = num_class
        self.fc_layer0 = nn.Linear(16, dim_h)
        self.fc_layer1 = nn.Linear(dim_h, dim_h // 2)
        self.fc_layer2 = nn.Linear(dim_h // 2, dim_h // 4)
        self.fc_layer3 = nn.Linear(dim_h // 4, dim_h // 4)
        self.fc_layer4 = nn.Linear(dim_h // 4, dim_h // 4)
        self.fc_layer5 = nn.Linear(dim_h // 4, dim_h // 4)
        self.fc_layer6 = nn.Linear(dim_h // 4, self.nclass)
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

    def forward(self, input_tensor):
        """
        :param
           - input_tensor: 输入的16维度的特征向量
        :return:
           - pred: 对 输入做分类,预测出的概率
        """
        y0 = self.fc_layer0(input_tensor)
        y0 = self.relu0(y0)
        y1 = self.dropout0(y0)
        # y1 的size是(batch_size,dim_h)
        y1 = self.fc_layer1(y1)
        y1 = self.relu1(y1)
        y2 = self.dropout1(y1)
        # y2 的size是(batch_size,dim_h//2)
        y2 = self.fc_layer2(y2)
        y2 = self.relu2(y2)
        y3 = self.dropout2(y2)
        # y3 的size是(batch_size,dim_h//4)
        y3 = self.fc_layer3(y3)
        y3 = self.relu3(y3)
        y4 = self.dropout3(y3)
        # y4 的size是(batch_size,dim_h//4)
        y4 = self.fc_layer4(y3)
        y4 = self.relu4(y4)
        y5 = self.dropout4(y4)
        # y5 的size是(batch_size,dim_h//4)
        y5 = self.fc_layer5(y5)
        y5 = self.relu5(y5)
        y6 = self.dropout5(y5)
        # y6 的size是(batch_size,dim_h//4)
        pred = self.fc_layer6(y6)
        return pred
