import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from generators import UNetGenerator
from discriminators import Discriminator
from losses import CategoryLoss, BinaryLoss
import os
from torch.optim.lr_scheduler import StepLR
from utils.init_net import init_net
import torchvision.utils as vutils
import torchvision.models as models

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier_biaszero(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

class ClassBlock_noDO(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock_noDO, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0)]
        if dropout:
            add_block += [nn.Dropout(p=0)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ClassBlock_mid(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=1024):
        super(ClassBlock_mid, self).__init__()

        mid_dim = num_bottleneck

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, mid_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier

        num_bottleneck = 512

        add_block_2 = []
        add_block_2 += [nn.Linear(mid_dim, num_bottleneck)]
        add_block_2 += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block_2 += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block_2 += [nn.Dropout(p=0.5)]
        add_block_2 = nn.Sequential(*add_block_2)
        add_block_2.apply(weights_init_kaiming)

        classifier_2 = []
        classifier_2 += [nn.Linear(num_bottleneck, class_num)]
        classifier_2 = nn.Sequential(*classifier_2)
        classifier_2.apply(weights_init_classifier)
        self.add_block_2 = add_block_2
        self.classifier_2 = classifier_2

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        x = self.add_block_2(x)
        x = self.classifier_2(x)
        return x


# Define mobileNet
class ft_mobile(nn.Module):

    def __init__(self):
        super(ft_mobile, self).__init__()
        model_ft = models.mobilenet_v3_small(pretrained=True)
        # avg pooling to global pooling
        model_ft.classifier = nn.Sequential()
        #model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft


    def forward(self, x):
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)

        # x = self.model.features[10](x)
        # x = self.model.features[11](x)
        # x = self.model.features[12](x)
        #x = self.model.avgpool(x)
        x = x.view(x.size()[0],-1)
        return x

# Define the ResNet50-based Model
class ft_net50(nn.Module):

    def __init__(self):
        super(ft_net50, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.layer4[0].downsample[0].stride = (1,)
        model_ft.layer4[0].conv2.stride = (1,1)
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        #x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size()[0],-1)
        return x

# Define ResNet-18 model
class ft_net18(nn.Module):

    def __init__(self):
        super(ft_net18, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.layer4[0].downsample[0].stride = (1,)
        model_ft.layer4[0].conv2.stride = (1,1)
        # self.classifier = nn.Linear(512,2)
        self.model = model_ft

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        #x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.avgpool(x)
        x = x.view(x.size()[0],-1)
        # x = self.classifier(x)
        return x

# define Zi2ZiModel (works recursively)
class Zi2ZiModel:
    def __init__(self, input_nc=3, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, n_layer=3):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.is_training = is_training
        self.image_size = image_size
        self.n_layer = n_layer

    def setup(self):

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            n_layer=self.n_layer
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            image_size=self.image_size
        )

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()

        if self.is_training:
            self.netD.train()
            self.netG.train()
        else:
            self.netD.eval()
            self.netG.eval()

    def set_input(self, labels, real_A, real_B):
        if self.gpu_ids:
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
            self.labels = labels.to(self.gpu_ids[0])
        else:
            self.real_A = real_A
            self.real_B = real_B
            self.labels = labels

    def forward(self):
        # generate fake_B

        self.fake_B, self.encoded_real_A = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)

    def backward_D(self, no_target_source=False):

        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB.detach())

        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty

        d_loss_real = self.real_binary_loss(real_D_logits)
        d_loss_fake = self.fake_binary_loss(fake_D_logits)

        self.d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        self.d_loss.backward()
        return category_loss

    def backward_G(self, no_target_source=False):

        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        # encoding constant loss
        # this loss assume that generated imaged and real image should reside in the same space and close to each other
        const_loss = self.Lconst_penalty * self.mse(self.encoded_real_A, self.encoded_fake_B)
        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * self.l1_loss(self.fake_B, self.real_B)
        fake_category_loss = self.Lcategory_penalty * self.category_loss(fake_category_logits, self.labels)

        cheat_loss = self.real_binary_loss(fake_D_logits)

        self.g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss
        self.g_loss.backward()
        return const_loss, l1_loss, cheat_loss

    def update_lr(self):
        # There should be only one param_group.
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_D learning rate from %.5f to %.5f." % (current_lr, update_lr))

        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_G learning rate from %.5f to %.5f." % (current_lr, update_lr))

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        category_loss = self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights

        # magic move to Optimize G again
        # according to https://github.com/carpedm20/DCGAN-tensorflow
        # collect all the losses along the way
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        const_loss, l1_loss, cheat_loss = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights
        return const_loss, l1_loss, category_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    net.load_state_dict(torch.load(load_path))
                else:
                    net.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))
                # net.eval()
        print('load model %d' % epoch)

    def sample(self, batch, basename):
        chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                label_dir = os.path.join(basename, str(label.item()))
                chk_mkdir(label_dir)
                vutils.save_image(image_tensor, os.path.join(label_dir, str(cnt) + '.png'))
                cnt += 1
            # img = vutils.make_grid(tensor_to_plot)
            # vutils.save_image(tensor_to_plot, basename + "_construct.png")
            '''
            We don't need generate_img currently.
            self.set_input(torch.randn(1, self.embedding_num).repeat(batch[0].shape[0], 1), batch[2], batch[1])
            self.forward()
            tensor_to_plot = torch.cat([self.fake_B, self.real_A], 3)
            vutils.save_image(tensor_to_plot, basename + "_generate.png")
            '''

class ft_zi2zi_Gen(nn.Module):

    def __init__(self, class_num=2):
        super(ft_zi2zi_Gen, self).__init__()
        self.layer1 = Downblock(1, 64, outermost=True)
        self.layer2 = Downblock(64, 128)
        self.layer3 = Downblock(128, 256)
        self.layer4 = Downblock(256, 512)
        self.layer5 = Downblock(512, 512)
        self.layer6 = Downblock(512, 512)
        self.layer7 = Downblock(512, 512)
        self.layer8 = Downblock(512, 512, innermost=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512,2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        # x = self.classifier(x)
        return x



class Downblock(nn.Module):
    def __init__(self, input_c, output_c, outermost=False, innermost=False):
        super(Downblock, self).__init__()
        if outermost:
            conv = nn.Conv2d(input_c, output_c, kernel_size=4,
                             stride=(2,2), padding=1, bias=False)
            down = [conv]
        elif innermost:
            relu = nn.LeakyReLU(0.2, True)

            conv = nn.Conv2d(input_c, output_c, kernel_size=4,
                             stride=(1,1), padding=1, bias=False)
            down = [relu, conv]
        else:
            relu = nn.LeakyReLU(0.2, True)

            conv = nn.Conv2d(input_c, output_c, kernel_size=4,
                             stride=(1,1), padding=1, bias=False)
            norm = nn.BatchNorm2d(output_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            down = [relu, conv, norm]

        self.down = nn.Sequential(*down)

    def forward(self, x):
        x = self.down(x)
        return x


def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size())
        return out

    def get_codes(self, x):
        return self.encoder(x)


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=5),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 3, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 1, kernel_size=5),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def get_codes(self, x):
        return self.encoder(x)



