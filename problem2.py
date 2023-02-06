import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Vi skal bare lage en klassifier
# DONE: Prøve å bruke Git og GitHub
# DONE: Implement accuracy in the training 
# TODO: Implement validation loss and accuracy in the training
# TODO: implement test performance hvertfall accuracy (kanskje confusion matrix og/eller den kurven) in training (ble anbefalt av stud.ass)
# TODO: Implement PR curves (final)
# TODO: Implement saving the results (trainings and validation accuracy and loss, test results)
    # and the code to a file (epochs, training/validation split, neural network structure, loss function, optimizer)
    # Document the files
# TODO: Implement confusion matrix

def validate(network, valloader, criterion):

  val_running_loss = 0.0

  #  defining model state
  network.eval()

  print('validating...')
  k=0

  with torch.no_grad(): #  preventing gradient calculations since we will not be optimizing
    #  iterating through batches
    for j, val_data in enumerate(valloader, 0):
      val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)

      #--------------------------
      #  making classsifications and computing loss
      #--------------------------
      val_outputs = net(val_inputs)
      val_loss = criterion(val_outputs, val_labels)
      val_running_loss += val_loss.item()
      k=i

  print("validation loss: ", val_running_loss/k)


def accuracy(network, dataloader):

  #  setting model state
  network.eval()
  
  #  instantiating counters
  total_correct = 0
  total_instances = 0

  with torch.no_grad(): #  preventing gradient calculations since we will not be optimizing
  #  iterating through batches
    for data in dataloader:
        images, labels = data[0].to(device), data[1].to(device)

        #-------------------------------------------------------------------------
        #  making classifications and deriving indices of maximum value via argmax
        #-------------------------------------------------------------------------
        classifications = torch.argmax(network(images), dim=1)

        #--------------------------------------------------
        #  comparing indicies of maximum values and labels
        #--------------------------------------------------
        correct_predictions = sum(classifications==labels).item()

        total_correct+=correct_predictions
        total_instances+=len(images)

    print(total_correct/total_instances)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])
device = "cuda:0" if torch.cuda.is_available() else "cpu"
kwargs = {} if device=='cpu' else {'num_workers': 1, 'pin_memory': True}
batch_size=10 # Kan endre denne? 

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

print(trainset)

train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2]) # kan endre på denne, hva som er fint og hjelper kommer an på kontekst ikke lett å si en enkel split (men fint å starte med 80/20)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)

valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                          shuffle=True, **kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, **kwargs)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module): # Kan endre de her
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

criterion = nn.CrossEntropyLoss() # Loss/distance funksjon
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # kan endre på denne

for epoch in range(5):  # loop over the dataset multiple times | Kan endre på denne?

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
    accuracy(net, trainloader)
    validate(net, valloader, criterion)
    accuracy(net, valloader)


           
    # Validation loss and accuracy?? (or too long to compute?)
    # also use accuracy?
    # Trainig loss and accuracy | validation loss and accuracy
    # good split between trainig and validation set

    # test accuracy every epoch
    # and accuracy
    # 80/20 er ok split, men burde endre hvis ikke får grei nok performance (vanligvis kommer det veldig an på problemet, ikke lett å si, må prøve)

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%'
      % (100 * correct / total))
