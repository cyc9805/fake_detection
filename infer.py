# from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode
import seaborn
from model import chk_mkdir

writer_dict = {
        '智永': 0, ' 隸書-趙之謙': 1, '張即之': 2, '張猛龍碑': 3, '柳公權': 4, '標楷體-手寫': 5, '歐陽詢-九成宮': 6,
        '歐陽詢-皇甫誕': 7, '沈尹默': 8, '美工-崩雲體': 9, '美工-瘦顏體': 10, '虞世南': 11, '行書-傅山': 12, '行書-王壯為': 13,
        '行書-王鐸': 14, '行書-米芾': 15, '行書-趙孟頫': 16, '行書-鄭板橋': 17, '行書-集字聖教序': 18, '褚遂良': 19, '趙之謙': 20,
        '趙孟頫三門記體': 21, '隸書-伊秉綬': 22, '隸書-何紹基': 23, '隸書-鄧石如': 24, '隸書-金農': 25,  '顏真卿-顏勤禮碑': 26,
        '顏真卿多寶塔體': 27, '魏碑': 28
    }


parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--run_all_label', action='store_true')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')
parser.add_argument('--type_file', type=str, default='type/宋黑类字符集.txt')


def draw_single_char(ch, font, canvas_size):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), ch, (0, 0, 0), font=font)
    img = img.convert('L')
    return img


def main(d_name, n_layer):
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    # chk_mkdir(infer_dir)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 모델 생성
    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False,
        n_layer=n_layer
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)
    model.netG.model.train(False)
    # 데이터 불러오기
    resize = (80,)
    transform_list = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            # transforms.Normalize((0, 0, 0), (1, 1, 1)),
            transforms.Normalize(0, 1),
        ])

    tx = transforms.Resize((80,), interpolation=InterpolationMode.BICUBIC)

    data_dir = '/Users/yongchanchun/Desktop/MacBook_Pro_Desktop/graduate_school/SDS_project/Dataset_date/' + d_name

    # image_datasets = {}
    image_datasets = datasets.ImageFolder(data_dir, transform_list)
    # image_datasets['typing'] = datasets.ImageFolder(os.path.join(data_dir, 'Dataset_typing'), transform_list)

    # dataloaders = {}
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle = False)
    # dataloaders['typing'] = torch.utils.data.DataLoader(image_datasets['typing'], batch_size=10)

    # temp의 값을 바꾸면 model의 값도 바뀜
    temp = model.netG.model
    #
    l_c = 0
    while True:
        l_c += 1
        temp.up = torch.nn.Sequential()
        if l_c > 6:
            temp.down[1].stride = (1, 1)
        if temp.innermost == True:
            break
        temp = temp.submodule

    features = torch.tensor([])
    labels = torch.tensor([])
    for i, data in enumerate(dataloader):
        images, label_ = data
        if images.shape[2] < 16:
            images = tx(images)
        out_feature = model.netG.model(images, count=0)
        features = torch.concat((features, out_feature), dim=0)
        labels = torch.concat((labels, label_), dim=0)

    features = features.detach().numpy()
    labels = labels.numpy()

    norm_ = np.linalg.norm(features, axis=1)
    norm_ = np.expand_dims(norm_, axis=1)
    features = features / norm_

    sim_matrix = np.dot(features, features.T)

    ax = seaborn.heatmap(sim_matrix)
    fig = ax.get_figure()
    fig.savefig(d_name+"_fig_zi2zi_last_2048_L"+str(n_layer)+'.png')


if __name__ == '__main__':
    with torch.no_grad():
        dataset_name = ['daechung', 'gangbuk_samsung']

        n_layer = 8
        for d_name in dataset_name:
            main(d_name, n_layer)

