import torch
from torch import nn
from torchvision.models.resnet import resnet50

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        resnet = resnet50(pretrained=True)
        loss_network = torch.nn.Sequential(*(list(resnet.children())[:-2])).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.counter = 0
        self.il = 0
        self.pl = 0
        self.al = 0

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        #Each epoch print out a running average of the individual losses too(!!!Need to update the value for the counter to match the number of batches per epoch)
        self.il += image_loss
        self.pl += perception_loss
        self.al += adversarial_loss
        if self.counter < 175:
            self.counter += 1
        else:
            print('Image: %.5f  Perception: %.5f  Advers: %.5f' % (self.il / 175, self.pl / 175, self.al / 175))
            self.counter = 1
            self.il = 0
            self.pl = 0
            self.al = 0
            
        return image_loss + 0.05 * perception_loss + 0.001 * adversarial_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
