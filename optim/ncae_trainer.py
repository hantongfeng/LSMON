from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import pdb


class NCAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001 ,gan_lr: float = 0.0002, std:float=1.0,idt:float=1.0,lamdba:float=0.1, n_epochs: int = 150, lr_milestones: tuple = (),spl:float=0.1,spm:float=0.7,
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0,normal_cls: int =0,known_outlier_class: int =0,ratio_pollution: int =0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        self.normal_cls = normal_cls
        self.lamdba=lamdba
        self.batch_size =batch_size
        self.std = std
        self.idt=idt
        self.gan_lr = gan_lr
        self.known_outlier_class=known_outlier_class
        self.ratio_pollution=ratio_pollution
        self.lr = lr
        self.spl=spl
        self.spm=spm
        self.topk = int(self.batch_size*0.0)
        self.split=int(self.batch_size*self.spl)
        self.split_max=int(self.batch_size*self.spm)
        self.mu = None
        self.mu_max=None
        self.inputs_old_max=None

    def train(self, dataset: BaseADDataset, net: BaseNet,d_l, d_s,d_g,g,d_k):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Initial setup for Adversarial learning
        real_label = 1
        fake_label = 0

        # Set loss
        criterion = nn.MSELoss(reduction='none')
        criterion_D = nn.BCELoss()
        criterion_D = nn.CrossEntropyLoss()

        # Set device
        net = net.to(self.device)
        netD_S = d_s.to(self.device)
        similar=d_g.to(self.device)

        if self.mu is None:
            logger.info('Initializing initial statistics...')
            self.mu = self.init_center_c(train_loader, net.encoder) #####################################
            self.mu_max=self.mu
            self.std_mtx = torch.ones(self.mu.size(),device=self.device)*self.std
            self.idt_mtx = torch.ones(self.mu.size(),device=self.device)*self.idt
            logger.info('Mu and Std initialized.')



        criterion = criterion.to(self.device)
        # inputs_old_max= torch.FloatTensor(int(self.batch_size-self.split_max),1, 28, 28).fill_(1).cuda()


        # Set optimizer (Adam optimizer for now)
        ##################### 
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer_d = optim.Adam(net.decoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
        optimizer_m = optim.Adam(similar.parameters(), lr=0.0005, betas=(0.5, 0.999))
        optimizer_e = optim.Adam(net.encoder.parameters(), lr=0.0005, betas=(0.5, 0.999))
        optimizer_s = optim.Adam(netD_S.parameters(), lr=0.0001, betas=(0.5, 0.999))
        #0.0005 = 88.32
        #0.0001 = 9

        # Set learning rate scheduler  
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=self.lr_milestones, gamma=0.1)
        scheduler_m = optim.lr_scheduler.MultiStepLR(optimizer_m, milestones=self.lr_milestones, gamma=0.1)
        scheduler_s = optim.lr_scheduler.MultiStepLR(optimizer_s, milestones=self.lr_milestones, gamma=0.1)
        scheduler_e = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train() # ?
        similar.train()
        netD_S.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            mu_lr = scheduler.get_last_lr()[0]
            for _s, data in enumerate(train_loader):

                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                gan_label = torch.LongTensor(inputs.size()[0]).fill_(0).cuda()
                similar_label = torch.LongTensor(self.split).fill_(0).cuda()
                similar_max_label = torch.LongTensor(int(self.batch_size-self.split_max)).fill_(0).cuda()

                ###########################
                # (3) Update Encoder network (f) + Generator (G) 
                ###########################
                with torch.no_grad():
                    rec,latent = net(inputs.detach(),get_latent=True)
                
                sort_loss = criterion(rec, inputs)
                _gamma = torch.mean(sort_loss, dim=tuple(range(1, rec.dim())))
                
                _, index_sorted = torch.sort(_gamma, dim=0, descending=False) 
                
                inputs_min=inputs[index_sorted[0:self.split]] 
                latent_min=latent[index_sorted[0:self.split]]
                rec_min=rec[index_sorted[0:self.split]]
                inputs_max=inputs[index_sorted[self.split_max:]] 
                latent_max=latent[index_sorted[self.split_max:]]
                rec_max=rec[index_sorted[self.split_max:]]

            
                if n_batches==0 and epoch==0:
                    self.inputs_old_max=inputs_max
                    self.inputs_old_min=inputs_min

                ###########################
                # (4) Update Encoder network (E) 
                ###########################
                optimizer_e.zero_grad()
                targetmin = Variable(similar_label.fill_(real_label))
                latent_zhong = net.encoder(inputs_min.detach())
                x_mu=self.mu.repeat(int(latent_zhong.size(0)),1)
                output_min = similar(latent_zhong.detach(),x_mu.detach()) 
                output_min = output_min.squeeze()
                errmin=criterion_D(output_min, targetmin)
                errmin.backward(retain_graph=True)
                
                targetmax = Variable(similar_label.fill_(fake_label))
                x_mumax=self.mu_max.repeat(int(latent_zhong.size(0)),1)
                output_max = similar(latent_zhong.detach(),x_mumax.detach()) 
                output_max = output_max.squeeze()
                errE =0.1-criterion_D(output_max, targetmax)
                errE.backward()
                optimizer_e.step()

                # ###########################
                # # (5) Update similar network   
                # ###########################
                for _ in range(latent_min.size()[0]):
                    if _==0:
                        noise = torch.normal(self.mu,self.std_mtx).view(1,-1)
                    else:
                        noise = torch.cat((noise,torch.normal(self.mu,self.std_mtx).view( 1,-1)),0)
                noise_mu = Variable(noise)

                for _ in range(latent_min.size()[0]):
                    if _==0:
                        noise = torch.normal(self.mu_max,self.idt_mtx).view(1,-1)
                    else:
                        noise = torch.cat((noise,torch.normal(self.mu_max,self.idt_mtx).view( 1,-1)),0)
                noise_mumax = Variable(noise)

                targetvr = Variable(similar_label.fill_(real_label))
                optimizer_m.zero_grad()
                output_min = similar(noise_mu.detach(),x_mu.detach()) 
                output_min = output_min.squeeze()
                errsimilar_min = criterion_D(output_min, targetvr)
                errsimilar_min.backward()

                targetvf = Variable(similar_label.fill_(fake_label))

                mumax=random.random()*(self.mu_max-self.mu)+self.mu_max
                mumax=mumax.repeat(int(latent_min.size(0)),1)
                output_max = similar(noise_mu.detach(),mumax.detach()) 
                output_max = output_max.squeeze()
                errsimilar_max = 0.1-criterion_D(output_max, targetvf)              
                errsimilar_max.backward()

                output_m = similar(noise_mumax.detach(),mumax.detach()) 
                output_m = output_m.squeeze()
                errsimilar_m = criterion_D(output_m, targetvr)
                errsimilar_m.backward()

                output_ = similar(noise_mumax.detach(),x_mu.detach()) 
                output_ = output_.squeeze()
                errsimilar_ = 0.1-criterion_D(output_, targetvf)
                errsimilar_.backward()

                optimizer_m.step()

                # ###########################
                # # (1) Update Generator network (G) 
                # ###########################
                optimizer_d.zero_grad()
                fake = net.decoder(noise_mu)
                targetv = Variable(similar_label.fill_(real_label)) 
                output = netD_S(fake)
                output = output.squeeze()
                errG = criterion_D(output, targetv)
                errG.backward()
                #errG_value = errG.item()
                optimizer_d.step()

                # ##########################
                # # (2) Update D_S network 
                # ##########################  
                targetv = Variable(similar_label.fill_(fake_label))
                optimizer_s.zero_grad()
                fakereal = net.decoder(noise_mu)
                output = netD_S(fakereal)
                output = output.squeeze()
                errD_S_real = 0.1-criterion_D(output, targetv)
                errD_S_real.backward()

                fake = net.decoder(latent_zhong) 
                targetv = Variable(similar_label.fill_(real_label))
                output = netD_S(fake.detach())
                output = output.squeeze()
                errD_S_fake = criterion_D(output, targetv)
                errD_S_fake.backward()
                #errD_S_value = errD_S_real.item() + errD_S_fake.item()
                optimizer_s.step()
                                              
                #####################################################

                optimizer.zero_grad()    
                rec,latent = net(inputs.detach(),get_latent=True)           
                total_loss = criterion(rec, inputs)
                total_loss = torch.mean(total_loss)
                total_loss.backward()
                optimizer.step()
                ###########################
                # (6) Update mu           
                ###########################

                
                old_rec1,old_latent1 = net(self.inputs_old_min,get_latent=True)
                rec_loss_min = criterion(rec_min, inputs_min)
                rec_loss2 = criterion(old_rec1, self.inputs_old_min) 
                base_loss1 = torch.cat((rec_loss2,rec_loss_min),0)
                scores1 = torch.mean(base_loss1, dim=tuple(range(1, rec.dim())))
                _, index_sorted1 = torch.sort(scores1, dim=0, descending=False)
                inputs_line1=torch.cat((self.inputs_old_min,inputs_min),0)
                letent_line1=torch.cat((old_latent1,latent_min),0)
                self.inputs_old_min=inputs_line1[index_sorted1[:int(latent_min.size(0))]]
                letent_l1=letent_line1[index_sorted1[:int(latent_min.size(0))]]
                self.mu = torch.mean(letent_l1,0)

                ###########################


                old_rec,old_latent = net(self.inputs_old_max,get_latent=True)
                rec_loss_max = criterion(rec_max, inputs_max)
                rec_loss = criterion(old_rec, self.inputs_old_max)
                base_loss = torch.cat((rec_loss,rec_loss_max),0)
                scores = torch.mean(base_loss, dim=tuple(range(1, rec.dim())))
                _, index_sorted = torch.sort(scores, dim=0, descending=True)
                inputs_line=torch.cat((self.inputs_old_max,inputs_max),0)
                letent_line=torch.cat((old_latent,latent_max),0)
                self.inputs_old_max=inputs_line[index_sorted[:int(latent_max.size(0))]]
                letent_l=letent_line[index_sorted[:int(latent_max.size(0))]]
                self.mu_max = torch.mean(letent_l,0)

                epoch_loss += total_loss.item()
                n_batches += 1

            scheduler.step()
            scheduler_d.step()
            scheduler_e.step()
            scheduler_m.step()
            scheduler_s.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            if epoch%5==0:
                print(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')


        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        return net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('Testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, _, idx = data
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('[Experimental results]---------------------------------------------------------------------------')
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')
        logger.info('================================================================================================')\
        
        with open('results_arryt.txt', 'a') as f:
            f.write(str('ratio_pollution is {:.2f}-->'.format(self.ratio_pollution)))
            f.write(str('split is {:.2f}-->'.format(self.spl)))
            f.write(str('split_max is {:.2f}-->'.format(self.spm)))
            f.write(str('std is {:.2f}-->'.format(self.std)))
            f.write(str('idt is {:.2f}-->'.format(self.idt)))
            # f.write(str('batch_size is {:.2f}-->'.format(self.batch_size)))
            f.write(str('Test AUC: {:.2f}%'.format(100. * self.test_auc)) + '\n')

    def init_center_c(self, train_loader: DataLoader, encoder: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data.
       """
        n_samples = 0
        c = torch.zeros(encoder.rep_dim, device=self.device)

        encoder.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = encoder(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
