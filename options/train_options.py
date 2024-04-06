from argparse import ArgumentParser

from configs.path_conf import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # general setup
        self.parser.add_argument('--exp', default='./exp', type=str,
                                 help='output directory')
        self.parser.add_argument('--dataset_type', default='CUHK', type=str,
                                 help='Type of dataset')
        self.parser.add_argument('--encoder_type', default='hifinet', type=str,
                                  help='Which encoder to use')
        self.parser.add_argument('--input_nc', default=6, type=int,
                                 help='Number of input image channels to the ifisketch network.')
        self.parser.add_argument('--output_size', default=1024, type=int,
                                 help='Output size of generator')
        self.parser.add_argument('--train_decoder', default=False, type=bool,
                                 help='Whether to train the decoder model')
        # batch size and dataloader works
        self.parser.add_argument('--batch_size', default=4, type=int,
                                 help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2, type=int,
                                 help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int,
                                 help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        # optimizers
        self.parser.add_argument('--learning_rate', default=0.0001, type=float,
                                 help='learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str,
                                 help='The optimizer used during training.')


        # loss lambdas
        self.parser.add_argument('--lpips_lambda', default=1, type=float,
                                 help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=1, type=float,
                                 help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1, type=float,
                                 help='L2 loss multiplier factor')


        # weights and checkpoint paths
        self.parser.add_argument('--stylegan_weights', default=model_paths["stylegan"], type=str,
                                 help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default=None, type=str,
                                 help='Path to Hifisketch model checkpoint')

        # intervals for logging, validation, and saving
        self.parser.add_argument('--max_steps', default=500000, type=int,
                                 help='Maximum number of training steps')
        self.parser.add_argument('--max_val_batches', type=int, default=None,
                                 help='Number of batches to run validation on. If None, run on all batches.')

        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int,
                                 help='Validation interval')
        self.parser.add_argument('--save_interval', default=None, type=int,
                                 help='Model checkpoint interval')

        # arguments for iterative encoding
        self.parser.add_argument('--n_iters_per_batch', default=1, type=int,
                                 help='Number of forward passes per batch during training')

        # hifinet parameters
        self.parser.add_argument('--load_w_encoder', action='store_true', help='Whether to load e4e encoder.')
        self.parser.add_argument('--w_encoder_checkpoint_path', default=model_paths["faces_encoder"], type=str,
                                 help='Path to pre-trained encoder.')
        self.parser.add_argument('--w_encoder_type', default='WEncoder',
                                 help='Encoder type for the encoder used to get the initial inversion')
        self.parser.add_argument('--layers_to_tune', default='0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24', type=str,
                                 help='comma-separated list of which layers of the StyleGAN generator to tune')

        self.parser.add_argument('--loss_mode', default='cos', type=str,
                                 help='loss mode')
        self.parser.add_argument('--patch_nums', default=256, type=int,
                                 help='nums of patches')
        self.parser.add_argument('--patch_size', default=32, type=int,
                                 help='size of patch')
        self.parser.add_argument('--use_norm', default=True, type=bool,
                                 help='whether to use norm')


    def parse(self):
        opts = self.parser.parse_args()
        return opts
