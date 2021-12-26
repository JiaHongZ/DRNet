# DRNet
This code is a implement of the paper "DRNet: A Deep Neuron Network With Multi-layer Residual Blocks Improves Image Denoising". 

This code is modified by https://github.com/cszn/KAIR.

Datasets:
-----------
Trainsets
- For Gaussian, Poisson, Salt and Pepper denoising on gray images and Gaussian denoising on color images

    [train400](https://github.com/cszn/DnCNN/tree/master/TrainingCodes/DnCNN_TrainingCodes_v1.0/data) 
    
    There are only 400 images in [train 400], which will result in frequently loading the dataset when using the big batch. So we flip and rotate these images to increase the number of images.
 
    [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)
    
    ImageNet
    
- For real noise
    
    [Trainsets from SIDD](http://130.63.97.225/sidd/dataset.php)

Testsets 
- For gray images

[set12](https://github.com/cszn/FFDNet/tree/master/testsets)

[bsd68](https://github.com/cszn/FFDNet/tree/master/testsets)

- For color images

McMaster

Kodak24

[cbsd68](https://github.com/cszn/FFDNet/tree/master/testsets)

- For real noise

Branchmark from SIDD(http://130.63.97.225/sidd/dataset.php)

Train
----------
main_train_###.py

- For Gaussian denoising on gray images: main_train_DRNet.py
- For Poisson denoising on gray images: main_train_DRNet_possion.py
- For Salt and Pepper denoising on gray images : main_train_DRNet_pepper.py
- For Gaussian denoising on color images : main_train_DRNet_color.py
- For real noise: main_train_DRNet_sidd.py

It is noted that DRNet mainly focuses on the Gaussian denoising. For Poisson and Salt and Pepper noise, we only use them when further comparing the residual blocks.

Testing
----------
main_test_###.py

The naming method of the test files is consistent with that of the training file.

For Gaussian noise, we provide pre-trained models of three noise level, 15, 25 and 50. Please ensure the test noise level is consistent with the model you select. Some names of the pre-trained models as follows:

---_gray15
---_gray25
---_gray50
---_color_15
---_color_25
---_color_50

To test the results of real denoising, you can run Submit.py to generate the result file,
SubmitSrgb.mat and submit it to http://130.63.97.225/sidd/benchmark_submit.php.

Pre-trained models:
-----------
Link：https://pan.baidu.com/s/1_FukDfK86-7jhzhphPN7MA 
Keyword：r11u

Directory structure
----------
- model_zoo (there are some pretrained model in the papar)
- options (Config files of the models)
- models (network, train method definition)
- model_zoo (pre-trained models)
- testsets
- trainsets
- utils
-----------
For more details， please read the Code Operation Guide.

Update operation
-----------
12.25  We solved one bug in this code.

12.26  We delete some redundant files for easily downloading and using. All resblocks we used are in the basicblock.py.

------------
If you meet this bug, please make sure that you are not running the code in Pycharm's test environment:
E:\soft2\anaconda\envs\pytorchgpu37\python.exe E:\PyCharm2020.1\plugins\python\helpers\pycharm\_jb_nosetest_runner.py --target test_DiehlAndCook2015.py::test
Traceback (most recent call last):
  File "E:\PyCharm2020.1\plugins\python\helpers\pycharm\_jb_nosetest_runner.py", line 4, in <module>
    import nose
ModuleNotFoundError: No module named 'nose'
