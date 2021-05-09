# DRNet
This code is a implement of the paper "DRNet: A Deep Neuron Network With Multi-layer Residual Blocks Improves Image Denoising". 

This code is modified by https://github.com/cszn/KAIR.

Datasets:
-----------
Trainsets
- For Gaussian, Poisson, Salt and Pepper denoising on gray images and Gaussian denoising on color images

    [train400](https://github.com/cszn/DnCNN/tree/master/TrainingCodes/DnCNN_TrainingCodes_v1.0/data)
 
    [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)
    
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

The file main_train_DRNet_###.py contains the best model in the paper: 
- For Gaussian denoising on gray images: main_train_DRNet.py
- For Poisson denoising on gray images: main_train_DRNet_possion.py
- For Salt and Pepper denoising on gray images : main_train_DRNet_pepper.py
- For Gaussian denoising on color images : main_train_DRNet_color.py
- For real noise: main_train_DRNet_sidd.py

Other files, such as main_train_DRNet_CCR.py, corresponds to the control model used in the ablation experiment.

Testing
----------
main_test_###.py

The naming method of the test files is consistent with that of the training file.

We condensed Possion and Salt & Pepper denoising test files into two files to reduce the number of files, main_test_pepper.py and main_test_possion.py.

To test the results of real denoising, you can run Submit.py to generate the result file,
SubmitSrgb.mat and submit it to http://130.63.97.225/sidd/benchmark_submit.php.

Pre-trained model:
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

------------
If you meet this bug, please make sure that you are not running the code in Pycharm's test environment:
E:\soft2\anaconda\envs\pytorchgpu37\python.exe E:\PyCharm2020.1\plugins\python\helpers\pycharm\_jb_nosetest_runner.py --target test_DiehlAndCook2015.py::test
Traceback (most recent call last):
  File "E:\PyCharm2020.1\plugins\python\helpers\pycharm\_jb_nosetest_runner.py", line 4, in <module>
    import nose
ModuleNotFoundError: No module named 'nose'
