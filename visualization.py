import numpy as np
import torch
import nibabel as nib
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

from networks.perspective_unet import PUnet


def show_label(image_, label_):
    img = image_.convert('RGBA')
    label_ = label_.convert('RGBA')
    x, y = img.size
    for i in range(x):
        for j in range(y):
            color = label_.getpixel((i, j))
            img_color = img.getpixel((i, j))
            Mean = int(np.mean(list(color[:-1])))

            colors = [img_color, (51, 68, 161, 255), (116, 204, 77, 255), (221, 46, 33, 255), (157, 227, 221, 255),
                      (182, 70, 174, 255), (235, 227, 52, 255), (106, 197, 225, 255), (242, 240, 233, 255)]
            color = colors[Mean]
            img.putpixel((i, j), color)
    return img


class FeatureVisual:
    def __init__(self, image, selected_layer):
        self.img = image
        self.selected_layer = selected_layer
        self.pretrain_model = PUnet(num_classes=9).cuda()
        self.pretrain_model.load_state_dict(torch.load("model_out_Synapse/epoch_599.pth"))

    def get_feature(self):
        x = self.img
        for index, layer in enumerate(self.pretrain_model.children()):
            x = layer(x)
            if index == self.selected_layer:
                return x


if __name__ == '__main__':
    # A basic example of visualizing the intermediate feature of the network.

    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(224), transforms.CenterCrop(224)])
    img_path = 'model_out_Synapse/predictions/case0035_img.nii.gz'  # You can change to your own root.
    gt_path = 'model_out_Synapse/predictions/case0035_gt.nii.gz'
    img_nii = nib.load(img_path)
    gt_nii = nib.load(gt_path)
    img_data = img_nii.get_fdata()
    gt_data = gt_nii.get_fdata()
    slice_idx = 74  # Choose which slice to show.
    data = img_data[:, :, slice_idx]
    label = gt_data[:, :, slice_idx]

    net_input = trans(data).unsqueeze(0)
    net_input = net_input.repeat(1, 3, 1, 1).to(torch.float).cuda()
    myClass1 = FeatureVisual(net_input, 0)
    myClass2 = FeatureVisual(net_input, 1)

    img_array1 = myClass1.get_feature()[0]
    img_array1 = F.interpolate(img_array1, size=(512, 512), mode='bicubic', align_corners=False).mean(dim=1)
    img_array1 = img_array1.squeeze(0).detach().cpu().numpy()
    img_array2 = myClass2.get_feature()
    img_array2 = F.interpolate(img_array2, size=(512, 512), mode='bicubic', align_corners=False).mean(dim=1)
    img_array2 = img_array2.squeeze(0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 1, figsize=(16, 16))
    im = axes.imshow(img_array1[:, :] - img_array2[:, :], cmap='jet', alpha=0.8)
    axes.imshow(data, cmap='gray', alpha=0.3)
    axes.axis('off')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Intensity')
    plt.savefig('heatmap.png')
    plt.show()

    image = Image.fromarray((data * 255).astype('uint8'))
    label = Image.fromarray(label)
    img = show_label(image, label)
    img.save("./gt.png")  # Path to save the heatmap figure.
