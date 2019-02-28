#coding:utf-8

import argparse

from data_providers.utils import get_data_provider_by_name
from models.alex_net import AlexNet
from models.simple_net import SimpleNet
from models.vgg16_net import VGG16Net


train_params_flavia = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.0001,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch': (150, 225),
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': 'every_epoch',  # shuffle dataset every epoch or not
    'normalization': 'by_chanels',
}

train_params_MK = {
    'batch_size': 64,
    'n_epochs': 1000,
    'initial_learning_rate': 0.001,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch': (200, 300, 400, 500),
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': 'every_epoch',  
    'normalization': 'by_chanels',
}

train_params_leaf198 = {
    'batch_size': 32,
    'n_epochs': 500,
    'initial_learning_rate': 0.001,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch': (30, 60, 100, 200, 300, 400),
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': 'every_epoch',  
    'normalization': 'by_chanels',
}

train_params_soyCultivar100 = {
    'batch_size': 32,
    'n_epochs': 500,
    'initial_learning_rate': 0.001,
    'reduce_lr_epoch_1': 150,
    'reduce_lr_epoch_2': 225,
    'reduce_lr_epoch': (100, 200, 300, 400),
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': 'every_epoch',
    'normalization': 'by_chanels',
}

def get_train_params_by_name(path):
    name = path.split(':')[-1]
    if name in ['flavia', 'flavia+']:
        return train_params_flavia
    if name in ['MK', 'MK+']:
        return train_params_MK
    if name in ['leaf198', 'leaf198+']:
        return train_params_leaf198
    if name in ['soyCultivar100', 'soyCultivar100+']:
        return train_params_soyCultivar100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['AlexNet', 'VGG16', 'SimpleNet'],
        default='VGG16',
        help='What type of model to use')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        help='What dataset should be used')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',
        default=1,
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)

    args = parser.parse_args()    # 解析参数

    # if not args.keep_prob:
    #     if args.dataset in ['C10', 'C100', 'SVHN', 'leaf']:
    #         args.keep_prob = 0.8
    #     else:
    #         args.keep_prob = 1.0
    # if args.model_type == 'DenseNet':
    #     args.bc_mode = False
    #     args.reduction = 1.0
    # elif args.model_type == 'DenseNet-BC':
    #     args.bc_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    train_params["save_path"] = "data"
    data_provider = get_data_provider_by_name(args.dataset, train_params)

    print("Initialize the model..")

    if args.model_type == 'VGG16':
        model = VGG16Net(data_provider=data_provider, **model_params)
    elif args.model_type == 'AlexNet':
        model = AlexNet(data_provider=data_provider, **model_params)
    elif args.model_type == 'SimpleNet':
        model = SimpleNet(data_provider=data_provider, **model_params)
    else:
        raise NotImplementedError
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=32)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))
