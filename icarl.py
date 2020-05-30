import math
import time
import torch
import numpy as np
from resnet_cifar import resnet32
import F
from copy import deepcopy

class iCaRL():
    def __init__(self, optimizer, scheduler, batch_size = 256, K = 20000):
        """
        :param K: total maximum number of exemplars (among all classes)
        """

        self.net = resnet32()
        
        self.num_observed_classes = 0
        self.exemplars = [] # list of classes, each one with a list of exemplars images for that class
        self.max_tot_exemplars = K
        self.batch_size = batch_size

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def classify(self, loader, DEVICE='cuda'):
        self = self.net.to(DEVICE)
        self.eval() # Sets the module in evaluation mode

        predictions = []
        for images, _ in loader:
            images = images.to(DEVICE)

            # Forward Pass
            outputs = self.forward(images)

            # Get predictions
            _, preds = torch.max(outputs.data, 1)
            predictions += preds

        return predictions
    
    def incremental_train(self, loader):
        """
        :param loader: loader with the new images
        """
        num_new_classes = self._get_new_classes_number(loader)
        
        update_representation(loader)
        m = int(self.K / (self.num_observed_classes + num_new_classes))

        # Remove exemplars from previous learned classes
        # [0, 1, ..., num_observed_classes - 1]
        for i in range(self.num_observed_classes):
            self.exemplars[i] = self.reduce_exemplar_set(self.exemplars[i], m)
        
        # Add exemplars from new classes
        # [num_observed_classes, ..., num_observed_classes + num_new_classes - 1]
        for i in range(self.num_observed_classes, self.num_observed_classes + num_new_classes):
            self.exemplars += [self.construct_exemplar_set(loader, m)]

        self.num_observed_classes += num_new_classes

    def reduce_exemplar_set(self, exemplars, m):
        """
        :param exemplars: list of exemplars of a specific class
        :param m: maximum number of exemplars for this class
        """
        max_num_exemplars_this_class = math.min(len(exemplars), m)
        return exemplars[:max_num_exemplars_this_class]

    def update_representation(self, loader, val_dataloader=None, verbose=False, device='cuda'):
        """
        :param loader: new data loader
        :param val_dataloader: compute accuracy on data from this dataloader
        :param verbose:
        :param device: 'cuda' (gpu) or 'cpu'
        """
        # form combined training set
        # D = new_images UNION exemplars
        num_classes_after_increment = num_observed_classes + self._get_new_classes_number(loader)


        num_current_exemplars = len(self.exemplars)
        total_loss = np.nan
        scheduler = self.scheduler
        old_net = deepcopy(self.net)

        # run network training (e.g. BackProp) with loss function
        for epoch in range(self.num_epochs):
            if verbose:
                print('Epoch {:>3}/{}\tLoss: {:07.4f}\tLearning rate: {}'.format(
                    epoch+1, self.num_epochs,
                    total_loss,
                    scheduler.get_last_lr()
                ))

            start_training_time = time.time()
            total_loss = 0.0
            total_training = 0
            for images, labels in loader:
                #images, labels = (images.to(device), labels.to(device))
                images, labels = (images.to(device), F.one_hot(labels, num_classes=self.num_classes).to(device, dtype=torch.float))

                self.train()

                # PyTorch, by default, accumulates gradients after each backward pass
                # We need to manually set the gradients to zero before starting a new iteration
                self.optimizer.zero_grad()

                # Forward pass to the network
                output = self.forward(images)

                
                # Output that would have gave the network before incrementing with new classes
                output_old = F.sigmoid(old_net(images))[:,:self.num_observed_classes]
                labels_onehot_new = F.one_hot(labels, num_classes_after_increment)
                target = torch.cat((output_old, labels_onehot_new), dim=1)

                loss = self.criterion(output, target)
                total_loss += loss.item() * len(labels)
                total_training += len(labels)

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer.step() # update weights based on accumulated gradients

            total_loss = total_loss / total_training
            epochs_stats[epoch] = {
                'loss': total_loss,
                'learning_rate': scheduler.get_last_lr(),
                'elapsed_time': time.time() - start_training_time
            }
            last_time = time.time()

            # Evaluate accuracy on validation set if verbose at each validation step
            if val_dataloader is not None:
                accuracy, _ = self.perform_test(val_dataloader)
                print(f'Epoch accuracy on validation set: {accuracy}')

            # Step the scheduler
            scheduler.step()
    
    def _get_new_classes_number(self, loader):
        labels = []
        for i, l in loader:
            labels += [l]
        unique_labels = np.unique(labels)
        return len(unique_labels)