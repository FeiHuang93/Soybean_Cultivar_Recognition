from .custom_provider_plus import CustomDataProvider


class Leaf198DataProvider(CustomDataProvider):
    """Abstract class for cifar readers"""
    # _data_url = "../data_jpg256"
    _n_classes = 198
    _data_shape = (256, 256, 3)
    # image, npz
    _file_type = "image"

    def __init__(self, data_url=None, save_path=None, validation_set=None, validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        super().__init__(data_url, save_path, validation_set, validation_split, shuffle, normalization, one_hot, **kwargs)


class Leaf198AugmentedDataProvider(Leaf198DataProvider):
    data_augmentation = True
