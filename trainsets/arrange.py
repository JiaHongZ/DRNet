# SIDD 将图片分为清晰图和噪声图
# 放在图片文件夹外
from PIL import Image
import os
import glob

def arrange(path = '', path1 = 'real', path2 = 'noise'):
    p = os.getcwd()
    path = os.path.join(p,path)
    path1 = os.path.join(p,path1)
    path2 = os.path.join(p,path2)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    filelist = os.listdir(path)
    for name in filelist:
        path_temp = os.path.join(path,name)
        clean_imgs = glob.glob(path_temp + '/*GT_SRGB*')
        noise_imgs = glob.glob(path_temp + '/*NOISY_SRGB*')
        for clean_img in clean_imgs:
            img = Image.open(clean_img,'r')

if __name__ == '__main__':
    arrange('Data')