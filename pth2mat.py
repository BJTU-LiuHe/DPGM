import os
import scipy.io
import torch as t

WILLOW_FEATURE_ROOT='data/cnn_features/willow'
PASCAL_FEATURE_ROOT='data/cnn_features/PascalVoc'

def _pth_to_mat(path) :
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isdir(filepath):
            _pth_to_mat(filepath)
        elif os.path.isfile(filepath):
            if os.path.basename(filepath).endswith('.pth'):
                infos = t.load(filepath)
                if infos['pts_features'] is None:
                    infos['pts_features'] = []

                portion, ext = os.path.splitext(filepath)
                savepath = portion + '.mat'
                print('savepath = ', savepath)
                scipy.io.savemat(savepath, infos)

if __name__ == '__main__':
    _pth_to_mat(PASCAL_FEATURE_ROOT)