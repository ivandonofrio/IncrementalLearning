import math
import time
import random
from numpy import linalg as LA
import numpy as np
from copy import deepcopy

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.backends import cudnn
from itertools import chain
from torch.autograd import Variable

# Classifiers
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# PyTorch
from torchvision import transforms
from tqdm import tqdm, tqdm_gui

# Losses
from .SNNLoss import SNNLoss

# GAN
from .ACGAN import ACGAN
from .cDCGAN import cDCGAN
from .WGAN import WGAN

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

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
        if not self.last:
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
		transform: torchvision transformations to apply to input data
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


class CosineLayer(nn.Module):

    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLayer, self).__init__()

        # Setup layer dimenstions
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        # Setup layer sigma parameter
        self.sigma = Parameter(torch.Tensor(1)) if sigma else None

        # Reset layer parameter
        self.reset_parameters()

    def reset_parameters(self):

        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)

        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, x):

        # Compute output
        x = F.linear(F.normalize(x), F.normalize(self.weight))

        # Scale by sigma if set
        if self.sigma is not None:
            x = self.sigma * x

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, parameters, use_exemplars, last_layer='std', num_classes=10, k=5000, loss='bce'):
        super(ResNet, self).__init__()
        """
        Make Incremental Learning ResNets great again!

        :param block:
        :param layers:
        :param parameters:
        :param use_exemplars: bool, store exemplars at the end of an iteration
		:param last_layer: string, 'cos' or 'std', use a cosine layer or a standard fully connected as last layer
        :param num_classes: int, number of initial classes
        """

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        if last_layer == 'cos':
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2, last_phase=True)
        else:
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        
        if last_layer == 'std':
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        elif last_layer == 'cos':
            self.fc = CosineLayer(64 * block.expansion, num_classes)
        else:
            raise(ValueError('Unknown last layer type'))

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

        if loss == 'snn':
            self.criterion = SNNLoss()
            self.snn_temperature = parameters['SNN_temperature']
        else:
            self.criterion = parameters['CRITERION']()

        self.loss = loss

        # Set utils structures
        self.use_exemplars = use_exemplars
        self.iterations = 0
        self.learned_classes = set()
        self.k = k
        self.processed_images = 0

		# Initialise GAN
        self.gan = WGAN(parameters['GAN_PARAMETERS'], self.num_classes)

        self.clf = {}   # cache classifiers object (SVM, KNN...) to test them
                        # multiple times without fitting it at each test
                        # (if no training in the meanwhile)
                        # key: 'svm' or 'knn', value: the fitted classifier

        # Exemplars structure
        self.exemplars = {}

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):

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

        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, get_only_features=False, get_also_features=False):
        """
        Give me an image and I will tell you what it could be
        (or how you can represent it if you pass me get_only_features=True)

        :param x: An image that you would never recognize
        :param get_only_features: bool, if False returns the output from the last FC layer
            if True returns output from the penultimate layer
		    :param get_also_features: bool, if True returns both
        """
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
        f = x.clone().detach()

        x = self.fc(x)

        if get_also_features:
            return x, f
        return x

    def perform_training(self, train_dataset, val_dataset=None, state_dict=None,
                         verbose=False, validation_step=5, classes_at_time=10, distillation=None,
                         policy='random', transform=None, classifier='fc', **classifier_kwargs):
        """
        Train me!

        :param train_dataset: the training dataset
        :param val_dataset: if not None, prints the accuracy on this dataset each validation_step epochs
        :param validation_steps: read this ^^^
        :param state dict: should I start from a pre-trained model? Ok, but please do not give me the ImageNet's one or you're a cheater
        :param verbose: bla bla bla
        :param classes_at_time: [not used anymore, but having useless thigs makes you giving more value to the other things]
		:param distillation: string, which distillation approach must be used, ['lwf','lfc']
        :param policy: string, ['random', 'norm']
        :param transform: the transformation to apply to the images of the dataset
        :param classifier: string, ['fc'] In future 'svm' could be added
        """
        # Setting up training framework
        torch.cuda.set_device(0)
        self = self.to(DEVICE)
        cudnn.benchmark
        current_classes = set()

        # Check if a previous state must be loaded
        if state_dict:
            self.load_state_dict(state_dict)

        # Store and freeze current network
        if distillation:
            old = deepcopy(self)
            for p in old.parameters():
                p.requires_grad = False

		      	# Parameters for less forget constraint
            if distillation == 'lfc' and self.iterations > 0:
                lmbd = 5 * ((len(self.learned_classes)/10) ** 0.5)
                K = 2
                print(f'Training with lambda {lmbd}')

        # Optimizer and scheduler setup
        optimizer = self.optimizer(self.parameters(), **self.optimizer_parameters)
        scheduler = self.scheduler(optimizer, **self.scheduler_parameters)

        # List of datasets to concatenate
        datasets_to_concatenate = []
        synt_dataset = []
        generated = []
        gen_labels = []
        gen_features = []

        # Generate and load training dataset
        datasets_to_concatenate.append(LabelledDataset(train_dataset, transform))

        new_classes = set([x[1] for x in train_dataset])
        if self.use_exemplars:

            # Merge new training image and exemplars
            exemplars_dataset = []

            for label in self.exemplars.keys():
                num_exemplars = len(self.exemplars[label]['exemplars'])
                for image in self.exemplars[label]['exemplars']:
                    exemplars_dataset.append((image, label))
            datasets_to_concatenate.append(LabelledDataset(exemplars_dataset, transform))

            dataset = ConcatDataset(datasets_to_concatenate)

            if self.iterations > 0:   
                # Generate synthetic features
                num_synt = max(0, 100 - num_exemplars)
                gen_features, gen_labels = self.gan.generate_examples(num_synt, list(self.learned_classes))
                # self.eval()
                # outputs = self.fc(gen_features.to(DEVICE))
                # _, preds = torch.max(outputs.data, 1)
                # print('{}% of generated features correctly classified'.format(torch.sum(preds == gen_labels.data).data.item() * 100 / len(gen_labels)))

        # Setting up data structures for statistics
        epochs_stats = {}
        last_time = time.time()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        print(f'Training on {len(loader.dataset)} images + {len(gen_labels)} synthetic features...')
        total_loss = math.nan

        if self.iterations > 0:
            num_gen_per_iter = (num_synt * len(self.learned_classes)) // len(loader)
            print(f'Take {num_gen_per_iter} features for each iter')

        self.gan.old_gen = deepcopy(self.gan.gen)
        self.gan.old_gen.eval()
        self.gan.gen.eval()

        epochs = range(self.num_epochs) if verbose else tqdm(range(self.num_epochs))
        for epoch in epochs:
            if verbose:
                print('Epoch {:>3}/{}\tLoss: {:07.4f}\tLearning rate: {}'.format(
                    epoch+1, self.num_epochs,
                    total_loss,
                    scheduler.get_last_lr()
                ))

            total_loss = 0.0
            total_training = 0

            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                target = F.one_hot(labels, num_classes=self.num_classes).to(DEVICE, dtype=torch.float)

                self.train()

                # PyTorch, by default, accumulates gradients after each backward pass
                # We need to manually set the gradients to zero before starting a new iteration
                optimizer.zero_grad()

                # Forward pass to the network
				        # Get also input features for cosine
                outputs, cur_features = self.forward(images, get_also_features=True)

                # Compute loss
                if distillation and self.iterations > 0:
                    # Store network outputs with pre-update parameters
                    old.eval()
                    output_old = old(images).to(DEVICE)

                    index = np.random.choice(range(len(gen_labels)), num_gen_per_iter, replace=False)
                    new_feat = gen_features[index].to(DEVICE)
                    new_label = gen_labels[index].type(torch.LongTensor).to(DEVICE)

                    new_gen = self.fc(new_feat)
                    old_gen = old.fc(new_feat)

                    # outputs = Variable(torch.cat((outputs, new_gen), dim=0), requires_grad=True)
                    outputs = torch.cat((outputs, new_gen), dim=0)
                    output_old = torch.cat((output_old, old_gen), dim=0)
                    y_onehot = torch.zeros(len(new_label), self.num_classes)
                    y_onehot.cuda().scatter_(1, new_label[:, None], 1)
                    # target = Variable(torch.cat((target, y_onehot.type(torch.FloatTensor).to(DEVICE))), requires_grad=True)
                    target = torch.cat((target, y_onehot.type(torch.FloatTensor).to(DEVICE)))
                    # target = torch.cat((target, F.one_hot(new_label.to(torch.int64), num_classes=self.num_classes).to(DEVICE, dtype=torch.float)))

                    target[:,list(self.learned_classes)] = nn.Sigmoid()(output_old[:,list(self.learned_classes)])

                loss = self.criterion(outputs, target)

                total_loss += loss.item() * len(labels)
                total_training += len(labels)

                loss.backward()
                optimizer.step()

                if epoch == 0:
                    with torch.no_grad():

                        # Store new classes and images
                        c = [l.item() for l in labels]
                        current_classes.update(c)

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

            # Store exemplars
            if self.use_exemplars:
                self.store_exemplars(train_dataset, policy=policy)

        # Reset all classifiers: the fitted ones are not valid anymore
        self.clf = {}

        if len(self.learned_classes) != self.num_classes:
            self.eval()
            self.gan.train(loader, self.learned_classes, self)
            print('Example of real features:', cur_features[0])

        self.iterations += 1

        return epochs_stats, (gen_features, gen_labels)

    def get_fittable_from_exemplars(self, exemplars = None):
        """
        Transform dict of exemplars in X and y, where X and y are fittable
            for scikit-learn classifiers

        :param exemplars: dict of exemplars, self.exemplars if None

        :return: X, y
        """
        if exemplars == None:
            exemplars = self.exemplars

        X = []  # List of features (one for each image)
        y = []  # List of labels
        
        # Convert dictionary of exemplars into:
        # - X: list of all images' representation
        # - y: list of all images' label
        for label, value in exemplars.items(): # For each label
            
            # Take representations of the exemplars of this label
            current_representations = value['representation']
            
            # Map each tensor to a numpy array (after putting it in CPU and detaching it from the graph)
            # and add them to the X list
            X += list(map(lambda tensor: tensor.cpu().detach().numpy(), current_representations))
            y += [label] * len(value['exemplars'])
        
        return X, y

    def get_predictions_from_classifier(self, images, clf):
        """
        Take the images, compute the features and predict them with the classifier

        :param images: images to predict
        :param clf: fitted classifier to use to predict

        :return: predictions
        """
        features = self.forward(images, get_only_features=True)

        # Bring tensor to CPU to transform it into a numpy array
        features = features.cpu().detach().numpy()
        normalized_features = list(map(lambda feature: feature / LA.norm(feature), features))

        preds = clf.predict(normalized_features)
        preds = torch.IntTensor(preds).to(DEVICE) # Convert to tensor and move to DEVICE

        return preds

    def perform_test(self, dataset, transform=None, classifier='fc', **classifier_kwargs):
        """
        :param classifier: string, ['fc', 'ncm', 'cos', 'svm', 'knn', 'rf'], where:
            - fc: last fully connected layer (standard ResNet)
            - ncm: nearest class mean
			- cos: cosine similarity
            - svm: SVM
            - knn: k-NN
            - rf: random forest
        """

        # If classifying with SVM or KNN, and that type of classifier is not cached yet
        if classifier in ['svm', 'knn', 'rf'] and classifier not in self.clf:
            X, y = self.get_fittable_from_exemplars()
            
            if classifier == 'svm':
                self.clf[classifier] = make_pipeline(StandardScaler(), SVC(**classifier_kwargs))

            elif classifier == 'knn':
                self.clf[classifier] = KNeighborsClassifier(**classifier_kwargs)

            elif classifier == 'rf':
                self.clf[classifier] = RandomForestClassifier(**classifier_kwargs)
            
            self.clf[classifier].fit(X, y)

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

                if classifier == 'ncm':
                    preds = self.get_nearest_classes(images)

                elif classifier == 'fc':
                    outputs = self.forward(images)
                    _, preds = torch.max(outputs.data, 1)
					
                elif classifier == 'cos':
                    preds = self.get_most_similar_cos(images)

                elif classifier in ['svm', 'knn', 'rf']:
                    preds = self.get_predictions_from_classifier(images, self.clf[classifier])

                else:
                    raise ValueError("Wrong value for argument 'classifier'")

                # Update Corrects
                correct_predictions += torch.sum(preds == labels.data).data.item()
                total_predictions += len(labels)

                prediction_history['pred'] += preds.tolist()
                prediction_history['true'] += labels.data.tolist()

        # Calculate Accuracy
        accuracy = correct_predictions / float(total_predictions)
        return accuracy, prediction_history

    def store_exemplars(self, images, policy='random'):
        """
        Select exemplars from the images and recompute all the others

        :param images: list of tuples (image, label), containing all the new images to add
        :param policy: string, ['random', 'norm', ...]
        """
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

                self.exemplars[label]['exemplars'] = selected_examplars
                self.exemplars[label]['tensors'] = selected_tensors
                self.exemplars[label]['representation'] = selected_representations
                self.exemplars[label]['mean'] = mean

                counter -= batch

    def get_mean_representation(self, exemplars):
        """
        :return: tuple (maps, mean), where:
            - maps: list of representations
            - mean: mean of all the representations
        """
        # Returns image features for current network and their non-normalized mean
        self.eval()

        # Extract maps from network
        with torch.no_grad():
            maps = [self.forward(exemplar.cuda().unsqueeze(0), get_only_features=True).cpu().detach().squeeze() for exemplar in exemplars]
            maps = [map/map.norm() for map in maps]

        return maps, torch.stack(maps).mean(0).squeeze()

    def get_nearest_classes(self, images):
        """
        Get nearest classes
        """
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
		
    def get_most_similar_cos(self, images):

        self.eval()
        with torch.no_grad():

            features = self.forward(images, get_only_features=True).detach()
            features = F.normalize(features)
            preds = []
            similarity = nn.CosineSimilarity(dim=0)

            for map in features:

                dst = {label:(similarity(map, self.exemplars[label]['mean'].to(DEVICE))) for label in self.exemplars.keys()}
                pred = max(dst, key=dst.get)
                preds.append(pred)

        return torch.Tensor(preds).to(DEVICE)

def resnet20(parameters, pretrained=False, use_exemplars=False, **kwargs):
    n = 3
    model = ResNet(BasicBlock, [n, n, n], parameters, use_exemplars, **kwargs)
    return model

def resnet32(parameters, pretrained=False, use_exemplars=False, **kwargs):
    n = 5
    model = ResNet(BasicBlock, [n, n, n], parameters, use_exemplars, **kwargs)
    return model

def resnet56(parameters, pretrained=False, use_exemplars=False, **kwargs):
    n = 9
    model = ResNet(Bottleneck, [n, n, n], parameters, use_exemplars, **kwargs)
    return model