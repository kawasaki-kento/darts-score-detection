import numpy as np
import pandas as pd
import random
import argparse
import sys
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter

parser = argparse.ArgumentParser() 
parser.add_argument("--input-data", type=str, default="", help="Feature data created with feature_creation.py")
parser.add_argument("--output-dir", type=str, default="", help="Output directory of the training model")
parser.add_argument("--split-percent", type=str, default="0.7", help="Percentage of training data after splitting")
parser.add_argument("--train-epochs", type=str, default="300", help="Number of learning epochs")
parser.add_argument("--onnx-option", type=bool, default=False, help="True if you want to output onnx files as well")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

if __name__ == '__main__':
    inputs = []
    labels = []

    with open(opt.input_data, 'r', newline='', encoding='utf-8') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for d in tsv_reader:
            # Four Inputs: relative_distance, relative_radian, bbox_width, bbox_height
            inputs.append([float(d[1]), float(d[2]), int(d[3]), int(d[4])])
            # Three Labels: Single, Double, and Triple
            labels.append(d[0].split(' ')[0])         
    assert(len(inputs) == len(labels))

    label_to_index = {}
    index_to_lable = {}
    for i, v in enumerate(set(labels)):
        label_to_index[v] = i
        index_to_lable[i] = v

    index_lables = [label_to_index[i] for i in labels]

    random.seed(27)
    data_index = [i for i in range(len(inputs))]
    random.shuffle(data_index)

    split_point = int(len(inputs)*float(opt.split_percent))
    train_index = data_index[:split_point]
    test_index = data_index[split_point:]

    train_inputs, test_inputs = [], []
    train_labels, test_labels = [], []

    for i in train_index:
        train_inputs.append(inputs[i])
        train_labels.append(index_lables[i])

    for i in test_index:
        test_inputs.append(inputs[i])
        test_labels.append(index_lables[i])

    assert(len(train_inputs) == len(train_labels))
    assert(len(test_inputs) == len(test_labels))
    print(f'{"="*25} Data {"="*25}')
    print(f'{"train_inputs:":15}{len(train_inputs)}', f'{"train_labels:":15}{len(train_labels)}')
    print(f'{"test_inputs:":15}{len(test_inputs)}', f'{"test_labels:":15}{len(test_labels)}')

    train_X = torch.Tensor(np.array(train_inputs))
    train_y = torch.LongTensor(np.array(train_labels))
    test_X = torch.Tensor(np.array(test_inputs))
    test_y = torch.LongTensor(np.array(test_labels))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(4, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.dp = nn.Dropout(p=0.3)
            self.fc3 = nn.Linear(64, len(label_to_index))
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.relu(self.fc2(x))
            x = self.dp(x)
            x = self.fc3(x)
            return F.log_softmax(x, dim = 1)

    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f'{"="*25} Train {"="*25}')
    model.train()
    for epoch in range(int(opt.train_epochs)+1):
        inp, tar = Variable(train_X), Variable(train_y)
        optimizer.zero_grad()
        output = model(inp)
            
        loss = F.nll_loss(output, tar)
        loss.backward()
        optimizer.step()
            
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(tar.data).sum().numpy() / len(train_X)
        
        if epoch % 100 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, loss.data.item(), accuracy))


    print(f'{"="*25} Eval {"="*25}')
    model.eval()

    predict_dct = {}
    c=0
    for i in range(len(test_inputs)):
        outputs = model(Variable(torch.Tensor([test_inputs[i]])))
        _, predicted = torch.max(outputs.data, 1)
        pred = predicted.numpy()[0]
        ans = test_labels[i]
        if pred == ans:
            c+=1
            predict_dct.setdefault(ans, 0)
            predict_dct[ans]+=1

    print(f'{"Accuracy:":21} '+'{:.3f}'.format(c/len(test_labels)))

    count = Counter(test_labels)
    for k, v in predict_dct.items():
        print(index_to_lable[k] + f'{" Accuracy:":15} '+'{:.3f}'.format(v/count[k]))

    torch.save(model.state_dict(), opt.output_dir+"dnn_model.pth")

    if opt.onnx_option:
        dummy_input = torch.randn(1,train_X.size()[1])
        torch.onnx.export(model, dummy_input, opt.output_dir+"dnn_model.onnx")