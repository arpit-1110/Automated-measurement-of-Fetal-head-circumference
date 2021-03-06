import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image
# import torchvision
from torchvision.transforms import ToPILImage
import time
from eval import eval_net
# import copy
from torch.autograd import Variable
from unet_model import UNet
from data_loader import HC18
import torch.nn.functional as F


def dice_coeff(inputs, target):
	eps = 1e-7
	coeff = 0
	# print(inputs.shape)
	for i in range(inputs.shape[0]):
		iflat = inputs[i,:,:,:].view(-1)
		tflat = target[i,:,:,:].view(-1)
		# print(torch.max(iflat), torch.min(iflat))
		# print(torch.max(tflat), torch.min(tflat))
		intersection = torch.dot(iflat, tflat)
		# print('intersection	', 2*intersection)
		# print(iflat.sum() + tflat.sum())
		coeff += (2. * intersection) / (iflat.sum() + tflat.sum() + eps)
	# print((2. * intersection) / (iflat.sum() + tflat.sum()))
	return coeff/inputs.shape[0]

def dice_loss(inputs, target):
	return 1 - dice_coeff(inputs, target)

train_set = HC18('train')
print('Train Set loaded')
val_set = HC18('val')
print('Validation Set loaded')
test_set = HC18('test')
print('Test Set loaded')


dataset = {0: train_set, 1: val_set}

dataloaders = {x: torch.utils.data.DataLoader(
    dataset[x], batch_size=2, shuffle=True, num_workers=0)for x in range(2)}
# print(dataloaders[0])

dataset_sizes = {x: len(dataset[x]) for x in range(2)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


def train_model(model, criterion, optimizer, scheduler=None, num_epochs=10):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch ' + str(epoch) + ' running')
        if epoch > 200:
            optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.0000005)
        elif epoch > 350:
            optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.0000005)
        elif epoch > 450:
            optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.0000005)
        for phase in range(2):
            if phase == 0:
                if scheduler:
                    scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            # running_corrects = 0
            # print(len(dataloaders[phase]))
            val_dice = 0
            count = 0
            for i, Data in enumerate(dataloaders[phase]):
            	count += 1
                inputs, masks = Data
                masks = masks/255
                # print('SHAPE', inputs.shape)
                inputs = inputs.to(device)
                masks = masks.to(device)
                inputs, masks = Variable(inputs), Variable(masks)
                # print(masks.shape)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 0):
                    pred_mask = model(inputs)
                    # pred_mask = (pred_mask-torch.min(pred_mask))/(torch.max(pred_mask) - torch.min(pred_mask))
                    # print(pred_mask.shape)
                    if not i % 4:
                        t = ToPILImage()
                        a = t(pred_mask[0].cpu().detach())
                        a.save('./Results/result_' + str(i) +
                               'epoch' + str(epoch) + '.png')
                    # print(pred_mask.shape)
                    if criterion is not None:
                        pred_mask = pred_mask.view(-1)
                        masks = masks.view(-1)
                        # print(pred_mask)

                        loss = criterion(pred_mask, masks)
                    else:
                        # print(masks.shape, pred_mask.shape)
                        # print(pred_mask)

                        # print(torch.max(pred_mask), torch.min(pred_mask))
                        loss = dice_loss(masks, pred_mask)
                        # print("loss in batch ",loss)
                    val_dice += dice_coeff(masks, pred_mask)
                    if phase == 0:
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item()
        epoch_loss = running_loss / dataset_sizes[phase]
        print('Epoch finished ! Loss: {}'.format(epoch_loss))
        # epoch_acc = running_corrects.double() / dataset_sizes[phase]
        # val_dice = 
        # val_dice = eval_net(model, val_set, torch.cuda.is_available())
        print('Training Dice Coeff: {}'.format(val_dice/count))

        print('End of epoch')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)
    return model


model = UNet(1, 1)
model = model.to(device)
criterion = nn.MSELoss()
model_optim = optim.SGD(model.parameters(), lr=0.00051, momentum=0.0000005)
# exp_lr_scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.1)
model = train_model(model, None, model_optim,
                    # exp_lr_scheduler,
                    num_epochs=500)
torch.save(model, './Models/model3')
