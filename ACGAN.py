import numpy as np
import math
import time

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

DEVICE = 'cuda'

def compute_acc(preds, labels):
    '''
    Computes the classification acc
    '''
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def normal_init(m, mean, std, has_bias=True):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if has_bias:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, d=384, c=3, num_classes=10, nz=100):
        super(Generator, self).__init__()
        self.d = d
        self.nz = nz
        self.num_classes = num_classes
        self.fc1 = nn.Linear(nz+num_classes, d)

        self.ct2 = nn.ConvTranspose2d(d, d//2, 4, 1, 0, bias=False)
        self.ct2_bn = nn.BatchNorm2d(d//2)

        self.ct3 = nn.ConvTranspose2d(d//2, d//4, 4, 2, 1, bias=False)
        self.ct3_bn = nn.BatchNorm2d(d//4)

        self.ct4 = nn.ConvTranspose2d(d//4, d//8, 4, 2, 1, bias=False)
        self.ct4_bn = nn.BatchNorm2d(d//8)

        self.ct5 = nn.ConvTranspose2d(d//8, c, 4, 2, 1, bias=False)

        # print(self)

    def forward(self, input):
        x = input.view(-1, self.nz + self.num_classes)
        x = self.fc1(x)
        x = x.view(-1, self.d, 1, 1)
        x = F.relu(self.ct2_bn(self.ct2(x)))
        x = F.relu(self.ct3_bn(self.ct3(x)))
        x = F.relu(self.ct4_bn(self.ct4(x)))
        x = torch.tanh(self.ct5(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std, False)


class Discriminator(nn.Module):
    def __init__(self, d=64, c=3, num_classes=10):
        super(Discriminator, self).__init__()
        self.d = d
        self.conv1 = nn.Conv2d(c, d, 3, 2, 1, bias=False)
        self.Drop1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(d, d*2, 3, 1, 1, bias=False)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.Drop2 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1, bias=False)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.Drop3 = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(d*4, d*8, 3, 1, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.Drop4 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(d*8, d*16, 3, 2, 1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(d*16)
        self.Drop5 = nn.Dropout(0.5)

        self.conv6 = nn.Conv2d(d*16, d*32, 3, 1, 1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(d*32)
        self.Drop6 = nn.Dropout(0.5)

        self.fc_dis = nn.Linear(4*4*d*32, 1)
        self.fc_aux = nn.Linear(4*4*d*32, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # print(self)

    def forward(self, img, get_features=False, T=1):
        x = self.Drop1(F.leaky_relu(self.conv1(img), 0.2))
        x = self.Drop2(F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2))
        x = self.Drop3(F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2))
        x = self.Drop4(F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2))
        x = self.Drop5(F.leaky_relu(self.conv5_bn(self.conv5(x)), 0.2))
        x = self.Drop6(F.leaky_relu(self.conv6_bn(self.conv6(x)), 0.2))

        #When d=16, d*32=512, TODO
        x = x.view(-1, 4*4*self.d*32)
        fc_aux = self.fc_aux(x)
        if get_features:
            return fc_aux
        fc_dis = self.fc_dis(x)
        liklihood_correct_class = self.softmax(fc_aux/T)
        liklihood_real_img = self.sigmoid(fc_dis).view(-1,1).squeeze(1)
        return liklihood_real_img, liklihood_correct_class

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std, False)


class ACGAN():
    def __init__(self, num_classes):
        self.num_classes = num_classes

        self.gen = Generator(num_classes=num_classes)
        self.gen.init_weights(mean=0.0, std=0.02)
        self.gen = self.gen.to(DEVICE)

        self.discr = Discriminator(num_classes=num_classes)
        self.discr.init_weights(mean=0.0, std=0.02)
        self.discr = self.discr.to(DEVICE)

        self.adv_criterion = torch.nn.BCELoss()
        self.aux_criterion = torch.nn.NLLLoss()
        # self.aux_criterion = torch.nn.CrossEntropyLoss()  

    def generate_examples(self, num_examples, active_classes, save=False, use_discr=False):
        '''
        Returns a dict[class] of generated samples.
        Just passing in random noise to the generator and storing the results in dict
        Generates a batch of 100 examples at a time
        num_examples: Total number of examples to generate
        active_classes: List of all classes trained on till now
        save: If True, also save samples of generated images to disk
        '''
        with torch.no_grad():
            # print("Note: Ignoring the fixed noise")
            self.gen.eval()
            #for param in G.parameters():
            #        param.requires_grad = False
            if use_discr:
                self.discr.eval()
            #if D is not None:
            #    for param in D.parameters():
            #        param.requires_grad = False
            examples = {}
            num_iter = 0
            for klass in active_classes:
                while ((not klass in examples.keys()) or (len(examples[klass]) < num_examples)):
                    print(f'Generating for class {klass}, iteration {num_iter}')
                    num_iter += 1

                    targets = np.zeros((10, self.num_classes))
                    targets[:, klass] = 1
                    nz_noise = np.random.normal(0, 1, (10, 100))
                    combined_noise = np.append(targets, nz_noise, axis=1)
                    noise = torch.from_numpy(combined_noise).to(DEVICE)
                    noise = noise.view(10, 100+self.num_classes, 1, 1).float()

                    images = self.gen(noise)

                    # if use_discr:
                    #     d_output = self.discr(images)
                    #     #Select imges that whose real-fake value > filter_val
                    #     #indices = (d_output[0] > args.filter_val).nonzero().squeeze()
                    #     #Select imges with P(img) belonging to class klass > filter_val
                    #     indices = (d_output[1][:, klass] > 0.2).nonzero().squeeze()
                    #     if indices.dim() == 0:
                    #         continue
                    #     images = torch.index_select(images, 0, indices)
                    if not klass in examples.keys():
                        examples[klass] = images
                    else:
                        examples[klass] = torch.cat((examples[klass],images), dim=0)

                # Dont save more than the required number of classes
                if save:
                    for i,img in enumerate(examples[klass]):
                        save_image(img, f'C{klass}img{i}.png')
                    
            # Trim extra examples
            if use_discr:
                for klass in active_classes:
                    examples[klass] = examples[klass][0:num_examples]
                print("[INFO] Examples matching the filter: ", len(active_classes) * (num_examples / num_iter), "%")

            self.gen.train()
            self.discr.train()
            return examples

    def train(self, loader, learned_classes):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.002, betas=(0.5, 0.999))
        gen_scheduler = optim.lr_scheduler.MultiStepLR(gen_opt, milestones=[20,40], gamma=0.1)
        
        discr_opt = torch.optim.Adam(self.discr.parameters(), lr=0.0002, betas=(0.5, 0.999))
        discr_scheduler = optim.lr_scheduler.MultiStepLR(discr_opt, milestones=[20,40], gamma=0.1)

        nz = self.num_classes + 100
        print("Start training GAN on classes {}".format(learned_classes))

        for epoch in range(50):
        # for epoch in range(2):
            acc_e = []
            a_losses_e = []
            d_losses_e = []
            g_losses_e = []
            fake_prob_e = []
            real_prob_e = []

            self.gen.train()
            self.discr.train()

            print("[GAN] Start Epoch {}\tGenerator LR:{}, Discriminator LR:{}".format(
                epoch + 1,
                gen_scheduler.get_last_lr(),
                discr_scheduler.get_last_lr()
            ))

            start = time.time()

            for images, labels in loader:
                batch_size = images.shape[0]

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                d_label = torch.FloatTensor(batch_size).to(DEVICE)

                ### Train Discriminator ###
                discr_opt.zero_grad()
                ## Train on real images
                d_label.data.fill_(1)

                d_output, a_output = self.discr(images)
                d_loss_real = self.adv_criterion(d_output, d_label)
                a_loss_real = self.aux_criterion(a_output, labels)
                d_real_total = d_loss_real + a_loss_real
                d_real_total.backward()
                real_prob_e.append(d_output.data.mean())
                acc_e.append(compute_acc(a_output, labels))

                ## Train on fake images
                # Random noise
                noise = torch.FloatTensor(batch_size, nz, 1, 1).to(DEVICE)
                noise.data.normal_(0, 1)
                # Noise in ACGAN consists of label info + noise
                nz_labels = np.random.choice(learned_classes, batch_size)
                nz_noise = np.random.normal(0, 1, (batch_size, 100))
                hot_labels = np.zeros((batch_size, self.num_classes))
                hot_labels[np.arange(batch_size), nz_labels] = 1

                combined_noise = np.append(hot_labels, nz_noise, axis=1)
                combined_noise = torch.from_numpy(combined_noise)

                noise.data.copy_(combined_noise.view(batch_size, nz, 1, 1))
                d_label.data.fill_(0)
                a_label = torch.from_numpy(nz_labels).to(DEVICE)

                g_output = self.gen(noise)
                g_output_temp = g_output.detach()
                d_output, a_output = self.discr(g_output_temp)
                d_loss_fake = self.adv_criterion(d_output, d_label)
                a_loss_fake = self.aux_criterion(a_output, a_label)
                d_fake_total = d_loss_fake + a_loss_fake
                d_fake_total.backward()
                d_loss_total = d_real_total + d_fake_total
                discr_opt.step()
                fake_prob_e.append(d_output.data.mean())
                d_losses_e.append(d_loss_total.cpu().data.numpy())


                ### Train Generator ###
                gen_opt.zero_grad()

                d_label.data.fill_(1)
                d_output, a_output = self.discr(g_output)
                d_loss_g = self.adv_criterion(d_output, d_label)
                a_loss_g = self.aux_criterion(a_output, a_label)
                g_loss_total = d_loss_g + a_loss_g
                g_loss_total.backward()
                gen_opt.step()
                g_losses_e.append(g_loss_total.cpu().data.numpy())

            gen_scheduler.step()
            discr_scheduler.step()

            # Stats
            time_taken = time.time() - start
            mean_g = (sum(g_losses_e)/len(g_losses_e))
            mean_d = (sum(d_losses_e)/len(d_losses_e))
            mean_acc = (sum(acc_e)/len(acc_e))
            mean_prob_real = (sum(real_prob_e)/len(real_prob_e))
            mean_prob_fake = (sum(fake_prob_e)/len(fake_prob_e))

            print("[GAN] Epoch: {}, g_loss: {}, d_loss: {}, acc: {}, D(x): {}, D(G(z)): {}, Time taken: {}".format(
                      epoch + 1, mean_g, mean_d, mean_acc,
                      mean_prob_real, mean_prob_fake, time_taken
                  ))