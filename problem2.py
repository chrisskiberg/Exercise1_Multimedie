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
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, average_precision_score, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
# auc=Area under curve

# https://github.com/bkong999/COVNet/blob/master/main.py


# DONE: Use Git and GitHub
# DONE: Implement accuracy in the training
# DONE: Implement confusion matrix
# DONE: Implement validation loss and accuracy in the training
# DONE: Implement PR curves
# ! TODO: Endre tilbake størrelsen på testset når ferdig å programmere funksionalitetene
# DONE: Micro average over all classes and plot the average PR curve
# DONE: Implement Area under the PR curve
# DONE: Implement precision and recall (training-, test- and validation set?)

# ? TODO: Implement saving function of the results (training and validation accuracy and loss, confusion matrix, PR curves)
# ! Will have to do this manually
# ? TODO continuation: And the code to a file (epochs, training/validation split, neural network structure, loss function, optimizer)
# ! Will have to do this manually


# TODO: Forstå alle funksjonene jeg skal bruke for forbedre modellen
# DONE: Clean up funksjoner, få god oversikt over tallene og plotene
# TODO: Trene modellen
# TODO: Klare å endre på modellen
# TODO: Trene modellene og lagre resultatene

def validate(network, valloader, criterion):

    val_running_loss = 0.0

    #  defining model state
    network.eval()

    print('validating...')
    k = 0

    with torch.no_grad():  # preventing gradient calculations since we will not be optimizing
        #  iterating through batches
        for j, val_data in enumerate(valloader, 0):
            val_inputs, val_labels = val_data[0].to(
                device), val_data[1].to(device)

            # --------------------------
            #  making classsifications and computing loss
            # --------------------------
            val_outputs = net(val_inputs)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
            k = j

    print("validation loss: ", val_running_loss/k)


def accuracy(network, dataloader, train_val_test=0):

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

        if train_val_test == 1:
            print("training accuracy: ", total_correct/total_instances)
        elif train_val_test == 2:
            print("validation accuracy: ", total_correct/total_instances)
        elif train_val_test == 3:
            print("test accuracy: ", total_correct/total_instances)
        else:
            print("[?] accuracy: ", total_correct/total_instances)


def precision_recall(y_pred, y_true):
    precision_arr = [[], [], [], [], [], [], [], [], [], []]
    recall_arr = [[], [], [], [], [], [], [], [], [], []]

    for x in range(len(precision_arr)):
        # En klasse er 1 og de andre er 0
        y_pred_class = []
        y_true_class = []
        for y in range(len(y_true)):
            if y_pred[y] == x:
                y_pred_class.append(1)
            else:
                y_pred_class.append(0)

            if y_true[y] == x:
                y_true_class.append(1)
            else:
                y_true_class.append(0)

        truePositives = 0
        trueNegatives = 0
        falsePositives = 0
        falseNegatives = 0

        for m in range(len(y_true)):  # Se figur for å dobbeltsjekke
            if y_pred_class[m] == y_true_class[m] and y_pred_class[m] == 1:
                truePositives += 1
            elif y_pred_class[m] == y_true_class[m] and y_pred_class[m] == 0:
                trueNegatives += 1
            elif y_pred_class[m] != y_true_class[m] and y_pred_class[m] == 1:
                falsePositives += 1
            elif y_pred_class[m] != y_true_class[m] and y_pred_class[m] == 0:
                falseNegatives += 1

        precision_class = truePositives / (truePositives+falsePositives)
        recall_class = truePositives / (truePositives+falseNegatives)

        precision_arr[x] = precision_class
        recall_arr[x] = recall_class

    return precision_arr, recall_arr


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device == 'cpu' else {'num_workers': 1, 'pin_memory': True}
batch_size = 10  # Kan endre denne?

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


# kan endre på denne, hva som er fint og hjelper kommer an på kontekst ikke lett å si en enkel split (men fint å starte med 80/20)
train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                        shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testset_80p, testset_20p = torch.utils.data.random_split(
    trainset, [0.98, 0.02])  # ! Må endres!!!!

testloader = torch.utils.data.DataLoader(testset_20p, batch_size=batch_size,
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

print("Starting training")
for epoch in range(2):  # loop over the dataset multiple times | Kan endre på denne?
    print("starting epoch " + str(epoch+1) + " ...")
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

    print('[%d] training running loss: %.3f' % (epoch + 1, running_loss / i))
    accuracy(net, trainloader, 1)
    validate(net, valloader, criterion)
    accuracy(net, valloader, 2)

print('-----------------')
print('Finished Training')
print('-----------------')

correct = 0
total = 0


y_pred = np.array([])
y_true = np.array([])
y_probs = np.array([], dtype="float")
y_prob = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        y_true = np.append(y_true, labels)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred = np.append(y_pred, predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_prob.append(probs.numpy().tolist())

print('Accuracy of the network: %d %%' % (100 * correct / total))


cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=[i for i in classes],
                     columns=[i for i in classes])
plt.figure(figsize=(12, 7))
sns.heatmap(df_cm, annot=True)
# plt.savefig('output.png')
plt.show()


y_prob_samples = []
# Det er 10 probabilities i et sample
# Det er 10 samples i 1 batch
# Det er 100 samples til sammen

y_true = np.array([int(x) for x in y_true])
# one hot encode the test data true labels
y_true_binary = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

precision_arr, recall_arr = precision_recall(y_pred, y_true)
print("Precision for each class: ", precision_arr)
print("Recall for each class: ", recall_arr)

for a in range(len(y_prob)):
    for b in range(len(y_prob[0])):
        y_prob_samples.append(y_prob[a][b])
# print(y_prob_samples)
y_prob_samples = np.array(y_prob_samples)
# Batch er 10, må dele opp slik at per sample og ikke per batch [[[...]]] ---> [[...]]
# Det er til sammen 1000 samples, og dette er også i y_prob selv om ser litt vanskelig ut


# Får riktig verdier, altså en verdi fra hver
# print(y_true_binary[:, 0])
# print()
# print(y_prob_samples[:, 0])

precision = dict()
recall = dict()
auc_precision_recall = []
average_precision = dict()
for i in range(len(classes)):
    precision[i], recall[i], _ = precision_recall_curve(
        y_true_binary[:, i], y_prob_samples[:, i])
    average_precision[i] = average_precision_score(
        y_true_binary[:, i], y_prob_samples[:, i])
    # test_precision[i], test_recall[i], _ = precision_recall_curve(y_test_binary[:, i], y_test_score[:, i])
    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

    auc_precision_recall.append(auc(recall[i], precision[i]))

print("AUPRC for each class: ", auc_precision_recall)
print("Average precision: ", average_precision)

plt.xlabel("recall")
plt.ylabel("precision")
plt.legend(loc="best")
plt.title("precision vs. recall curve")
plt.show()

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_true_binary.ravel(), y_prob_samples.ravel()
)
average_precision["micro"] = average_precision_score(
    y_true_binary, y_prob_samples, average="micro")

# auc_precision_recall.append(auc(recall[i], precision[i]))


display = PrecisionRecallDisplay(
    recall=recall["micro"],
    precision=precision["micro"],
    average_precision=average_precision["micro"],
)
display.plot()
_ = display.ax_.set_title("Micro-averaged over all classes")
plt.show()

# Tar man AUCPR fra baseline eller fra x aksen?

display_x = display.line_.get_xdata()
display_y = display.line_.get_ydata()
display_auc = auc(display_x, display_y)
print("display_auc: ", display_auc)


# print(display.line_.get_xdata())
# print(display.line_.get_xydata())
# print(display.line_.get_ydata())


# print(display._y)
# print(display._xy)


print("hei")
