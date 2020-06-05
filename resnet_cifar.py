import math
import time
import random

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.backends import cudnn
from torchvision import transforms

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


class LabelledDataset(Dataset):
    '''Custom dataset for labelled images.

    Arguments:
        data (list of tuples (image, label)): list of labelled images
    '''
    def __init__(self, data, transform=None):
        super(LabelledDataset).__init__()
        self.images = []
        self.labels = []
        for x in data:
            self.images.append(x[0])
            self.labels.append(x[1])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ResNet(nn.Module):

    def __init__(self, block, layers, parameters, lwf, use_exemplars, ncm, num_classes=10, k=5000):
        super(ResNet, self).__init__()

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
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

        # Hyperparameters
        self.num_classes = num_classes
        self.batch_size = parameters['BATCH_SIZE']
        self.num_epochs = parameters['NUM_EPOCHS']
        self.scheduler = parameters['SCHEDULER']
        self.scheduler_parameters = parameters['SCHEDULER_PARAMETERS']
        self.optimizer = parameters['OPTIMIZER']
        self.optimizer_parameters = parameters['OPTIMIZER_PARAMETERS']
        self.criterion = parameters['CRITERION']()

        # Set utils structures
        self.lwf = lwf
        self.use_exemplars = use_exemplars
        self.ncm = ncm
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

    def forward(self, x, get_only_features=False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if get_only_features:
            return x

        x = self.fc(x)
        return x

    def perform_training(self, train_dataset, val_dataset=None, state_dict=None, verbose=False, validation_step=5, classes_at_time=10, policy='random', transform=None):

        # Setting up training framework
        self = self.to(DEVICE)
        cudnn.benchmark
        current_classes = set()

        # Setting up data structures for statistics
        epochs_stats = {}
        last_time = time.time()

        # Check if a previous state must be loaded
        if state_dict:
            self.load_state_dict(state_dict)

        # Store and freeze current network
        if self.lwf:
            old = deepcopy(self)
            for p in old.parameters():
                p.requires_grad = False

        # Optimizer and scheduler setup
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        scheduler = self.scheduler(optimizer, **self.scheduler_parameters)

        # Generate and load training dataset
        dataset = LabelledDataset(train_dataset, transform)
        if self.use_exemplars:

            # Merge new training image and exemplars
            exemplars_dataset = []

            for label in self.exemplars.keys():
                for image in self.exemplars[label]['exemplars']:
                    exemplars_dataset.append((image, label))

            dataset = ConcatDataset([dataset, LabelledDataset(exemplars_dataset, transform)])

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        print(f'Training on {len(loader)*self.batch_size} images...')

        for epoch in range(self.num_epochs):
            if verbose:

                print('Epoch {:>3}/{}\tLoss: {:07.4f}\tLearning rate: {}'.format(
                    epoch+1, self.num_epochs,
                    total_loss if len(epochs_stats) > 0 else -1,
                    scheduler.get_last_lr()
                ))

            total_loss = 0.0
            total_training = 0

            for images, labels in loader:

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

                if epoch == 0:
                    with torch.no_grad():

                        # Store new classes and images
                        c = [l.item() for l in labels]
                        current_classes.update(c)

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer.step() # update weights based on accumulated gradients

            # Store loss
            total_loss = total_loss/total_training

            # Update statistics
            epochs_stats[epoch] = {
                'loss': total_loss,
                'learning_rate': scheduler.get_last_lr(),
                'elapsed_time': time.time() - last_time
            }

            last_time = time.time()

            # Evaluate accuracy on validation set if verbose at each validation step
            if val_dataset and verbose and (epoch + 1) % validation_step == 0:
                accuracy, _ = self.perform_test(val_dataset)
                print(f'Epoch accuracy on validation set: {accuracy}')

            # Step the scheduler
            scheduler.step()

        # Update learned classes
        with torch.no_grad():

            self.learned_classes.update(current_classes)
            self.iterations += 1

            # Store exemplars
            if self.use_exemplars:
                self.store_exemplars(train_dataset, policy=policy)

        return epochs_stats

    def perform_test(self, dataset, transform=None):

        self = self.to(DEVICE)
        with torch.no_grad():

            # Network in evaluation mode
            self.eval()

            # Generate dataloader from current dataset
            dataloader = DataLoader(
                LabelledDataset(dataset, transform),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )

            correct_predictions = 0
            total_predictions = 0
            prediction_history = {
                'pred': [],
                'true': [],
            }

            for images, labels in dataloader:
                images, labels = (images.to(DEVICE), labels.to(DEVICE))

                if self.ncm:
                    preds = self.get_nearest_classes(images)
                else:
                    outputs = self.forward(images)
                    _, preds = torch.max(outputs.data, 1)

                # Update Corrects
                correct_predictions += torch.sum(preds == labels.data).data.item()
                total_predictions += len(labels)

                prediction_history['pred'] += preds.tolist()
                prediction_history['true'] += labels.data.tolist()

        # Calculate Accuracy
        accuracy = correct_predictions / float(total_predictions)
        return accuracy, prediction_history

    def store_exemplars(self, images, policy='random'):

        self.eval()

        with torch.no_grad():

            # Handle dimension
            new_classes = []

            # Collect incoming images and labels as (PIL, label) tuple
            incoming_data = images
            self.processed_images += len(incoming_data)

            # Compute bound and class batch size
            bound = counter = min(self.k, self.processed_images)
            batch = bound // len(self.learned_classes)
            print(f'Storing {batch} exemplars per class...')

            # Store new PIL images and labels
            for image, label in incoming_data:

                if label not in self.exemplars:
                    self.exemplars[label] = {
                        'mean': None,
                        'exemplars': [],
                        'tensors': [],
                        'representation': []
                    }
                    new_classes.append(label)

                # Convert to tensor and normalize
                tensor = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])(image)

                # Store current image and tensor
                self.exemplars[label]['exemplars'].append(image)
                self.exemplars[label]['tensors'].append(tensor)

            for label in self.exemplars.keys():

                # Get current mean and features
                features, mean = self.get_mean_representation(self.exemplars[label]['tensors'])
                mean /= torch.norm(mean)

                # Store only m exemplars with different policies
                if policy == 'random':

                    # Get random indices sample
                    index_sample = random.sample(list(range(batch)), min(batch, counter))

                    # Random subset of exemplars, representation and tensors
                    selected_examplars = [el for i, el in enumerate(self.exemplars[label]['exemplars']) if i in index_sample]
                    selected_tensors = [el for i, el in enumerate(self.exemplars[label]['tensors']) if i in index_sample]
                    selected_representations = [el for i, el in enumerate(features) if i in index_sample]

                elif policy == 'norm':

                    if label in new_classes:

                        # Store class exemplars
                        current_exemplars = self.exemplars[label]['exemplars'].copy()
                        current_tensors = self.exemplars[label]['tensors'].copy()
                        current_representations = features.copy()

                        # Initialise features and exemplars subset collection
                        selected_examplars = []
                        selected_tensors = []
                        selected_representations = []

                        # Setup incremental features collector
                        incremental_features_sum = torch.FloatTensor(len(features[0]) * [0])

                        while len(selected_examplars) < batch and len(current_exemplars) > 0:

                            # Store norms from current mean
                            norms = []

                            # Associate each representation to its distance from mean
                            for index, image in enumerate(current_exemplars):

                                # Sum current image and
                                feature = current_representations[index]

                                scaled_features_sum = (feature + incremental_features_sum)/(len(selected_examplars) + 1)
                                scaled_features_sum /= scaled_features_sum.norm()

                                # Get norm of difference
                                diff_norm = (mean - scaled_features_sum).norm()
                                norms.append(diff_norm)

                            # Get index of min distance
                            index = norms.index(min(norms))

                            # Update selection with nearest exemplar and representation
                            selected_examplars.append(current_exemplars[index])
                            selected_tensors.append(current_tensors[index])
                            selected_representations.append(current_representations[index])

                            # Update representation sum
                            incremental_features_sum += selected_representations[-1]

                            # Remove elements from representations and exemplars sets
                            del current_representations[index], current_exemplars[index], current_tensors[index]

                    else:

                        # If not new class only select best representations
                        selected_examplars = self.exemplars[label]['exemplars'][:batch]
                        selected_tensors = self.exemplars[label]['tensors'][:batch]
                        selected_representations = features[:batch]

                elif policy == 'clustering':

                    if label in new_classes:

                        from sklearn.cluster import DBSCAN, AffinityPropagation
                        import numpy as np

                        # Store class exemplars
                        current_exemplars = self.exemplars[label]['exemplars'].copy()
                        current_tensors = self.exemplars[label]['tensors'].copy()
                        current_representations = features.copy()

                        # Each tensor as input of the algorithm
                        # Each tensor as input of the algorithm
                        X = [tensor.cpu().numpy() for tensor in current_representations]

                        #cluster = DBSCAN(eps=.2, min_samples=3, n_jobs=4).fit(X)
                        cluster = AffinityPropagation().fit(X)

                        #print(len(cluster.cluster_centers_))

                        best_representative = list(map(lambda i, x: (i, [np.linalg.norm(x - center) for center in cluster.cluster_centers_]), X, range(len(X))))
                        #best_representative = list(zip(range(len(X)), best_representative))
                        best_representative = [(min(values), index) for values, index in best_representative]
                        sorted(best_representative, key=lambda pair: pair[1])

                        #print(best_representative)

                        indices = [index for value, index in best_representative][:batch]
                        print(indices)

                        #print(indices)

                        selected_examplars = [el for i, el in enumerate(self.exemplars[label]['exemplars']) if i in indices]
                        selected_tensors = [el for i, el in enumerate(self.exemplars[label]['tensors']) if i in indices]
                        selected_representations = [el for i, el in enumerate(features) if i in indices]

                        print(len(selected_examplars), len(selected_tensors), len(selected_representations))

                    else:

                        # If not new class only select best representations
                        selected_examplars = self.exemplars[label]['exemplars'][:batch]
                        selected_tensors = self.exemplars[label]['tensors'][:batch]
                        selected_representations = features[:batch]

                    """external_points = 0
                    valid_points = 0
                    for index, lbl in enumerate(cluster.labels_):

                        labels_dict[lbl].append(index)

                        if lbl != -1:
                            valid_points += 1
                        else:
                            external_points += 1

                    #print(external_points, valid_points)
                    # Generate collection of images
                    collection = [(lbl, indices) for lbl, indices in labels_dict.items()]
                    sorted(collection, key=lambda pair: len(pair[1]), reverse=True)
                    #print(collection)

                    iter = 0
                    indices = []
                    while len(indices) < batch and valid_points + external_points > 0:

                        index = iter % len(collection)
                        lbl, elements = collection[index]

                        if lbl != -1 and valid_points > 0:

                            if len(elements) > 0:
                                indices.append(elements.pop())
                                valid_points -= 1

                            iter += 1

                        elif lbl == -1 and valid_points <= 0:
                            indices.append(elements.pop())
                            external_points -= 1
                        else:
                            iter += 1

                    print(len(indices), indices)"""

                    # Initialise features and exemplars subset collection
                    #selected_examplars = [el for i, el in enumerate(self.exemplars[label]['exemplars']) if i in indices]
                    #selected_tensors = [el for i, el in enumerate(self.exemplars[label]['tensors']) if i in indices]
                    #selected_representations = [el for i, el in enumerate(features) if i in indices]

                self.exemplars[label]['exemplars'] = selected_examplars
                self.exemplars[label]['tensors'] = selected_tensors
                self.exemplars[label]['representation'] = selected_representations
                self.exemplars[label]['mean'] = mean

                counter -= batch

    def get_mean_representation(self, exemplars):

        # Returns image features for current network and their non-normalized mean
        self.eval()

        # Extract maps from network
        with torch.no_grad():
            maps = [self.forward(exemplar.cuda().unsqueeze(0), get_only_features=True).cpu().detach().squeeze() for exemplar in exemplars]
            maps = [map/map.norm() for map in maps]

        return maps, torch.stack(maps).mean(0).squeeze()

    def get_nearest_classes(self, images):

        self.eval()
        with torch.no_grad():

            features = self.forward(images, get_only_features=True).detach()
            features = F.normalize(features)
            preds = []

            for map in features:

                dst = {label: (map - self.exemplars[label]['mean'].to(DEVICE)).pow(2).sum() for label in self.exemplars.keys()}
                pred = min(dst, key=dst.get)
                preds.append(pred)

        return torch.Tensor(preds).to(DEVICE)

def resnet20(parameters, pretrained=False, lwf=False, use_exemplars=False, ncm=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], parameters, lwf, use_exemplars, ncm, **kwargs)
    return model

def resnet32(parameters, pretrained=False, lwf=False, use_exemplars=False, ncm=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], parameters, lwf, use_exemplars, ncm, **kwargs)
    return model

def resnet56(parameters, pretrained=False, lwf=False, use_exemplars=False, ncm=False, **kwargs):
    n = 9
    model = ResNet(Bottleneck, [n, n, n], parameters, lwf, use_exemplars, ncm, **kwargs)
    return model
