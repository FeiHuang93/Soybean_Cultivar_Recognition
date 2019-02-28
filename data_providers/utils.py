#coding:utf-8

from .flavia_provider import FlaviaDataProvider, FlaviaAugmentedDataProvider
from .leaf198t_provider import Leaf198DataProvider,Leaf198AugmentedDataProvider
from .MK_provider import MKDataProvider, MKAugmentedDataProvider
from .custom_provider_plus import CustomDataProvider,CustomAugmentedDataProvider
from .SoyCultivar100_provider import Soy100AugmentedDataProvider


def get_data_provider_by_name(path, train_params):
    """Return required data provider class"""
    name = path.split(':')[-1]     # 从路径中与数据名的组合中分离出数据名
    train_params['data_url'] = path.split(':')[0]       # 分离出路径
    if name == 'custom':
        return CustomDataProvider(**train_params)
    if name == 'custom+':
        return CustomAugmentedDataProvider(**train_params)
    if name == 'flavia':
        return FlaviaDataProvider(**train_params)
    if name == 'flavia+':
        return FlaviaAugmentedDataProvider(**train_params)
    if name == 'MK':
        return MKDataProvider(**train_params)
    if name == 'MK+':
        return MKAugmentedDataProvider(**train_params)
    if name == 'leaf198':
        return Leaf198DataProvider(**train_params)
    if name == 'leaf198+':
        return Leaf198AugmentedDataProvider(**train_params)
    if name == 'soyCultivar100+':
        return Soy100AugmentedDataProvider(**train_params)
    else:
        print("Sorry, data provider for `%s` dataset "
              "was not implemented yet" % name)
        exit()
