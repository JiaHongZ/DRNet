

'''
# --------------------------------------------
# select dataset
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# --------------------------------------------
'''


def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['drnet', 'denoising']:
        from data.dataset_drnet import DatasetDRNet as D
    elif dataset_type in ['sidd', 'denoising']:
        from data.dataset_sidd import DatasetSIDD as D
    elif dataset_type in ['drnet_pepper', 'denoising']:
        from data.dataset_drnet_pepper import DatasetDRNet_SP as D
    elif dataset_type in ['drnet_possion', 'denoising']:
        from data.dataset_drnet_possion import DatasetDRNet_Possion as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
