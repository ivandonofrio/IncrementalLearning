import math
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.backends import cudnn

DEVICE = 'cuda'

"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, parameters, lwf, num_classes=10, k=5000):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
		
        # hyperparameters
        self.num_classes = num_classes
        self.num_epochs = parameters['NUM_EPOCHS']
        self.scheduler = parameters['SCHEDULER']
        self.scheduler_parameters = parameters['SCHEDULER_PARAMETERS']
        self.optimizer = parameters['OPTIMIZER']
        self.optimizer_parameters = parameters['OPTIMIZER_PARAMETERS']
        self.criterion = parameters['CRITERION']()

        # Set utils structures
        self.lwf = lwf
        self.iterations = 0
        self.learned_classes = set()
        self.k = k
        self.processed_images = 0

        # Exemplars structure
        self.exemplars = {}

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
		
    def perform_training(self, train_dataloader, val_dataloader=None, state_dict=None, verbose=False, validation_step=5, classes_at_time=10):

        self = self.to(DEVICE)
        cudnn.benchmark
        current_classes = set()

        if state_dict:
            self.load_state_dict(state_dict)

        # Store and freeze current network
        if self.lwf:
            old = deepcopy(self)
            for p in old.parameters():
                p.requires_grad = False
        
        epochs_stats = {}
        last_time = time.time()
        
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        scheduler = self.scheduler(optimizer, **self.scheduler_parameters)

        training_images = []
        training_classes = []
        
        for epoch in range(self.num_epochs):
            if verbose:
                print('Epoch {:>3}/{}\tLoss: {:07.4f}\tLearning rate: {}'.format(
                    epoch+1, self.num_epochs,
                    total_loss if len(epochs_stats) > 0 else -1,
                    scheduler.get_last_lr()
                ))

            total_loss = 0.0
            total_training = 0
            for images, labels in train_dataloader:
                images = images.to(DEVICE)
                target = F.one_hot(labels, num_classes=self.num_classes).to(DEVICE, dtype=torch.float)

                self.train()

                # PyTorch, by default, accumulates gradients after each backward pass
                # We need to manually set the gradients to zero before starting a new iteration
                optimizer.zero_grad()

                # Forward pass to the network
                outputs = self.forward(images)

                # Compute loss
                if self.lwf and self.iterations > 0:
                    # Store network outputs with pre-update parameters
                    with torch.no_grad():
                        old.eval()
                        output_old = old(images).to(DEVICE)

                    # Include old predictions for distillation
                    target[:,list(self.learned_classes)] = nn.Sigmoid()(output_old[:,list(self.learned_classes)])
				
                loss = self.criterion(outputs, target)
                total_loss += loss.item() * len(labels)
                total_training += len(labels)

                # Store new classes and images
                c = [l.item() for l in labels]

                training_images += [image.data for image in images]
                training_classes += c
                current_classes.update([l.item() for l in labels])

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer.step() # update weights based on accumulated gradients

            total_loss = total_loss/total_training
            epochs_stats[epoch] = {
                'loss': total_loss,
                'learning_rate': scheduler.get_last_lr(),
                'elapsed_time': time.time() - last_time
            }
            last_time = time.time()

            # Evaluate accuracy on validation set if verbose at each validation step
            if val_dataloader and verbose and (epoch + 1) % validation_step == 0:
                accuracy, _ = self.perform_test(val_dataloader)
                print(f'Epoch accuracy on validation set: {accuracy}')

            # Step the scheduler
            scheduler.step()

		# Update learned classes
        self.learned_classes.update(current_classes)
        self.iterations += 1

        # Store exemplars
        self.store_exemplars(training_classes, training_images)
        
        return epochs_stats
      
    def perform_test(self, dataloader):
        self = self.to(DEVICE)
        
        with torch.no_grad():
            self.eval() # Sets the module in evaluation mode

            correct_predictions = 0
            total_predictions = 0
            prediction_history = {
                'pred': [],
                'true': [],
            }

            for images, labels in dataloader:
                images, labels = (images.to(DEVICE), labels.to(DEVICE))

                # Forward Pass
                outputs = self.forward(images)

                # Get predictions
                _, preds = torch.max(outputs.data, 1)

                # Update Corrects
                correct_predictions += torch.sum(preds == labels.data).data.item()
                total_predictions += len(labels)

                prediction_history['pred'] += preds.tolist()
                prediction_history['true'] += labels.data.tolist()

        # Calculate Accuracy
        accuracy = correct_predictions / float(total_predictions)

        return accuracy, prediction_history

    def store_exemplars(self, classes, images):

        # Handle dimensions 
        incoming_data = list(zip(images, classes))
        self.processed_images += len(incoming_data)

        bound = counter = min(self.k, self.processed_images)
        batch = round(bound/len(self.learned_classes))

        for image, label in incoming_data:

            if label not in self.exemplars:
                self.exemplars[label] = {
                    'mean': None,
                    'exemplars': []
                }

            self.exemplars[label].exemplars.append(image)

        for label in self.exemplars.keys():

            self.exemplars[label]['exemplars'] = random.sample(self.exemplars[label]['exemplars'], min(batch, count))
            self.exemplars[label]['mean'] = torch.mean(torch.stack(self.exemplars[label]['exemplars']), 0, keepdim=True)
            counter -= batch


def resnet20(parameters, pretrained=False, lwf=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], parameters, lwf, **kwargs)
    return model

def resnet32(parameters, pretrained=False, lwf=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], parameters, lwf, **kwargs)
    return model

def resnet56(parameters, pretrained=False, lwf=False, **kwargs):
    n = 9
    model = ResNet(Bottleneck, [n, n, n], parameters, lwf, **kwargs)
    return model