from .custom_provider_plus import CustomDataProvider


class FlaviaDataProvider(CustomDataProvider):
    """Abstract class for cifar readers"""
    _data_url = "../flavia"
    _n_classes = 33
    _data_shape = (224, 224, 3)
    # image, npz
    _file_type = "image"

    def __init__(self, save_path=None, validation_set=None, validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        super().__init__(save_path, validation_set, validation_split, shuffle, normalization, one_hot, **kwargs)


class FlaviaAugmentedDataProvider(FlaviaDataProvider):
    data_augmentation = True
