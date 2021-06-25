import torch
import torch.nn as nn
from torch.nn.functional import softmax


class ClassifierModel(nn.Module):
    """

    ClassifierModel:
    The eigenvector is first entered into the first fully connected layer: fc_layer0,
    then activated by relu and dropout once,tensor size will be 1024
    then　enter to the second full connection layer fc_layer1,tensor size will be 512
    ...

    Finally it will become a tensor which size is`self.nclass` ,It's the probability of each of the categories

    """

    def __init__(self, num_class, dropout, dim_h=1024):
        """
        :arg:
        num_class Number of classes to classify
        dropout  Perform random forgetting ratios for some of the input dimensions to prevent overfitting
        dim_h The dimension of the first hidden layer
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
           - input_tensor: 16-dimensional eigenvectors of the input
        :return:
           - pred: classify the inputs and predict the probabilities
        """
        y0 = self.fc_layer0(input_tensor)
        y0 = self.relu0(y0)
        y1 = self.dropout0(y0)
        # y1 size is (batch_size,dim_h)
        y1 = self.fc_layer1(y1)
        y1 = self.relu1(y1)
        y2 = self.dropout1(y1)
        # y2 size is (batch_size,dim_h//2)
        y2 = self.fc_layer2(y2)
        y2 = self.relu2(y2)
        y3 = self.dropout2(y2)
        # y3 size (batch_size,dim_h//4)
        y3 = self.fc_layer3(y3)
        y3 = self.relu3(y3)
        y4 = self.dropout3(y3)
        # y4 size (batch_size,dim_h//4)
        y4 = self.fc_layer4(y3)
        y4 = self.relu4(y4)
        y5 = self.dropout4(y4)
        # y5 size (batch_size,dim_h//4)
        y5 = self.fc_layer5(y5)
        y5 = self.relu5(y5)
        y6 = self.dropout5(y5)
        # y6 size (batch_size,dim_h//4)
        pred = self.fc_layer6(y6)
        return pred
