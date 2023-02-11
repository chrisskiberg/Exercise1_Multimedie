import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchmetrics.classification import PrecisionRecallCurve

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
# auc=Area under curve

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.preprocessing import OneHotEncoder


# https://github.com/bkong999/COVNet/blob/master/main.py


# DONE: Use Git and GitHub
# DONE: Implement accuracy in the training
# DONE: Implement confusion matrix
# DONE: Implement validation loss and accuracy in the training
# ! TODO: Implement PR curves (final)
# ! TODO: Endre tilbake størrelsen på testset når ferdig å programmere funksionalitetene
# TODO: Implement saving function of the results (training and validation accuracy and loss, confusion matrix, PR curves)
# TODO continuation: And the code to a file (epochs, training/validation split, neural network structure, loss function, optimizer)

# def plot_precision_recall_curve(dataloader, _classifier, caller):# hva er x, y og caller??

#     # # put y into multiple columns for OneVsRestClassifier
#     # onehotencoder = OneHotEncoder()

# #  For each classifier, the class is fitted against all the other classes
#     # y_pred = clf.predict(X_test)
#     # y_proba = clf.predict_proba(X_test)

#     # Compute ROC curve and ROC area for each class
#     fig = plt.figure()
#     plt.style.use('default')
#     precision = dict()
#     recall = dict()
#     for i in range(n_classes):
#         precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_proba[:, i])
#         plt.plot(recall[i], precision[i], lw=2, label='PR Curve of class {}'.format(i))

#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("recall")
#     plt.ylabel("precision")
#     plt.legend(loc="lower right", prop={'size': 10})
#     plt.title('Precision-Recall to multi-class: ' + caller)
#     # plt.suptitle(algor_name, fontsize=16)
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     plt.show()


def validate(network, valloader, criterion):

    val_running_loss = 0.0

    #  defining model state
    network.eval()

    print('validating...')
    k = 0

    with torch.no_grad():  # preventing gradient calculations since we will not be optimizing
        #  iterating through batches
        for j, val_data in enumerate(valloader, 0):
            val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)

            # --------------------------
            #  making classsifications and computing loss
            # --------------------------
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
            k = j

    print("validation loss: ", val_running_loss/k)


def accuracy(network, dataloader):

    #  setting model state
    network.eval()

    #  instantiating counters
    total_correct = 0
    total_instances = 0

    with torch.no_grad():  # preventing gradient calculations since we will not be optimizing
        #  iterating through batches
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            # -------------------------------------------------------------------------
            #  making classifications and deriving indices of maximum value via argmax
            # -------------------------------------------------------------------------
            classifications = torch.argmax(network(images), dim=1)

            # --------------------------------------------------
            #  comparing indicies of maximum values and labels
            # --------------------------------------------------
            correct_predictions = sum(classifications == labels).item()

            total_correct += correct_predictions
            total_instances += len(images)

        print(total_correct/total_instances)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device == 'cpu' else {'num_workers': 1, 'pin_memory': True}
batch_size = 10  # Kan endre denne?

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

print(trainset)

# kan endre på denne, hva som er fint og hjelper kommer an på kontekst ikke lett å si en enkel split (men fint å starte med 80/20)
train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testset_90p, testset_10p = torch.utils.data.random_split(trainset, [0.98, 0.02]) # ! Må endres!!!!

testloader = torch.utils.data.DataLoader(testset_10p, batch_size=batch_size,
                                         shuffle=False, **kwargs)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):  # Kan endre de her
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()  # Loss/distance funksjon
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9)  # kan endre på denne

for epoch in range(1):  # loop over the dataset multiple times | Kan endre på denne?

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    print('[%d] loss: %.3f' % (epoch + 1, running_loss / i))
    # accuracy(net, trainloader)
    # validate(net, valloader, criterion)
    # accuracy(net, valloader)


print('Finished Training')

correct = 0
total = 0


# ---------------- HERE TOP

y_pred = np.array([])
y_true = np.array([])
y_probs = np.array([],dtype="float")

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        y_true=np.append(y_true, labels)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred=np.append(y_pred, predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        probs = torch.nn.functional.softmax(outputs, dim=1)
        # y_prob.extend(probs.detach().cpu().numpy())
        y_probs=np.append(y_probs, probs.detach().cpu().numpy())
        # print(y_true)
        # print(data)

        # y_prob = np.concatenate(y_prob)
    # for X_test, y_test in testloader:
    #   print("lol")

print('Accuracy of the network on the 10000 test images: %d %%'
      % (100 * correct / total))

# jeg mener labels er ground truth

# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
#                      columns=[i for i in classes])
# plt.figure(figsize=(12, 7))
# sns.heatmap(df_cm, annot=True)
# # plt.savefig('output.png')
# plt.show()

# net.eval()
# logits = net(testloader.data)
# y_proba = F.softmax(logits, dim=1) # assuming logits has the shape [batch_size, nb_classes]
# preds = torch.argmax(logits, dim=1)


precision = dict()
recall = dict()
fig = plt.figure()
plt.style.use('default')


# Skille klassene (gjøre hver klasse til binær (one vs rest))
# Alle som har samme klasse får 1 (pos label)? og de andre får 0
# ---------
y_true = np.array([ int(x) for x in y_true ])
print("y_true: ", y_true)
y_true_sort_index = np.argsort(y_true, kind="stable") # Ascending
print("Indicies: ", y_true_sort_index)
y_true_sorted=np.sort(y_true, kind="stable") # Ascending
print("Sorted array: ", y_true_sorted)

# ! One vs rest per klasse

onevsrest_arr=[[],[],[],[],[],[],[],[],[],[]] # 0,1,2,3,4,5,6,7,8 og 9, hvor indexsene er i hver liste 
for i in range(len(y_true_sorted)):
    onevsrest_arr[y_true_sorted[i]].append(y_true_sort_index[i])
print(onevsrest_arr)
print("hei")

# for i in range(len(classes)):  
#     precision[i], recall[i], _ = precision_recall_curve(
#         y_true[:, i], y_probs[:, i])
#     plt.plot(recall[i], precision[i], lw=2,
#              label='PR Curve of class {}'.format(i))


# print(np.argwhere(y_true_sorted==0))
# print(np.argwhere(y_true_sorted==1))
# print(np.argwhere(y_true_sorted==2))

# ---------

# ifølge dokumentasjonen er det mulig å lage en multiclass, men da må jeg oppgi multiclass selv, og kan bli enda mer stress
# https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier
# for i in range(len(classes)):  
#     precision[i], recall[i], _ = precision_recall_curve(
#         y_true[:, i], y_probs[:, i])
#     plt.plot(recall[i], precision[i], lw=2,
#              label='PR Curve of class {}'.format(i))

# # ---------------- HERE BOTTOM




















# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel("recall")
# # plt.ylabel("precision")
# # plt.legend(loc="lower right", prop={'size': 10})
# # plt.title('Precision-Recall to multi-class: ')
# # # plt.suptitle(algor_name, fontsize=16)
# # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# # plt.show()

# # pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=len(classes))
# # precision, recall, thresholds = pr_curve(y_pred, y_true)
# # print(precision)
# # print(recall)
# # print(thresholds)
