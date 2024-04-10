import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_label(image, label):
    img = image.convert('RGBA')
    label = label.convert('RGBA')
    x, y = img.size
    for i in range(x):
        for j in range(y):
            color = label.getpixel((i, j))
            img_color = img.getpixel((i, j))
            Mean = int(np.mean(list(color[:-1])))

            colors = [img_color, (51, 68, 161, 255), (116, 204, 77, 255), (221, 46, 33, 255), (157, 227, 221, 255),
                      (182, 70, 174, 255), (235, 227, 52, 255), (106, 197, 225, 255), (242, 240, 233, 255)]
            color = colors[Mean]
            img.putpixel((i, j), color)
    return img


if __name__ == '__main__':
    # Loading NII File
    img_path = '/home/zxk/Perspective+/model_out_Synapse/predictions/case0004_img.nii.gz'
    pred_path = '/home/zxk/Perspective+/model_out_Synapse/predictions/case0004_pred.nii.gz'
    gt_path = '/home/zxk/Perspective+/model_out_Synapse/predictions/case0004_gt.nii.gz'

    img_nii = nib.load(img_path)
    pred_nii = nib.load(pred_path)
    gt_nii = nib.load(gt_path)

    # Convert to Numpy
    img_data = img_nii.get_fdata()
    pred_data = pred_nii.get_fdata()
    gt_data = gt_nii.get_fdata()

    slice_idx = 76  # Choose the slice you want to show

    image = Image.fromarray((img_data[:, :, slice_idx] * 255).astype('uint8'))
    image.save("./image.png")
    label = Image.fromarray(pred_data[:, :, slice_idx])
    img = show_label(image, label)
    img.save("./pred.png")
    image = Image.fromarray((img_data[:, :, slice_idx] * 255).astype('uint8'))
    label = Image.fromarray(gt_data[:, :, slice_idx])
    img = show_label(image, label)
    img.save("./gt.png")

    plt.show()
