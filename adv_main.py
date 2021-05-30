# python examples/pretnormal_training_robustness.py --dataset mushrooms --tensorboard

import os
import sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../')
import math
import common.experiments
import common.utils
import common.eval
import numpy
from common.log import log
from common.paths import DATA_DIR_TRAIN
# import models
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train_utils import *
from adversarial_training import *

import gc


def find_incomplete_state_file(model_file):
    """
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])


class Main:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.directory = './.assets/checkpoints'
        """ Arguments of program. """


        self.trainloader = None
        """ (torch.utils.data.DataLoader) Training loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """

        self.adversarialloader = None
        """ (torch.utils.data.DataLoader) Loader to attack. """

        self.epsilon = 0
        """ (float) Epsilon for L_inf attacks. """


    def setup(self):
        """
        Set dataloaders.
        """
        self.epsilon = 0.03

        batch_size = 12
        image_transforms = {
        #     # Train uses data augmentation
            'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=299),#, scale=(1., 1.0)
                transforms.RandomPerspective(distortion_scale=0.45, p=0.4),
                # transforms.RandomRotation(degrees=15),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(size=400),
                transforms.Resize(size=224),
                transforms.ColorJitter(0.3, 0.35, 0.3, 0.04),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        #     # Validation does not use augmentation
            'val':
            transforms.Compose([
                # transforms.Resize(size=(350)),
                # transforms.CenterCrop(299),
                transforms.CenterCrop(size=400),
                transforms.Resize(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        dataloaders = get_dataloaders(DATA_DIR_TRAIN, 0.3,  batch_size, image_transforms)
        self.trainloader = dataloaders['train']
        self.testloader = dataloaders['val']
        self.adversarialloader = dataloaders['val']

        common.utils.makedir(self.directory)

    def get_training_attack(self):
        """
        Get attack for training.
        """

        epsilon = 0.03
        ace = attacks.AdvCF()
        ace.num_classes = 2
        ace.search_steps = 1
        ace.max_iterations = 40
        ace.initial_const = 8
        ace.device = torch.device("cuda")
        ace.pieces = 64
        ace.norm = attacks.norms.LInfNorm()
        objective = attacks.objectives.UntargetedF0Objective()

        return ace, objective


    def get_test_attacks(self):
        """
        Get attacks to test.
        """

        pgd = attacks.BatchGradientDescent()
        pgd.max_iterations = 200
        pgd.base_lr = 0.005
        pgd.momentum = 0.9
        pgd.c = 0
        pgd.lr_factor = 1.25
        pgd.normalized = True
        pgd.backtrack = True
        pgd.initialization = attacks.initializations.LInfUniformNormInitialization(self.epsilon)
        pgd.projection = attacks.projections.SequentialProjections([
            attacks.projections.LInfProjection(self.epsilon),
            attacks.projections.BoxProjection()
        ])
        pgd.norm = attacks.norms.LInfNorm()
        untargetedf0 = attacks.objectives.UntargetedF0Objective()

        return [
            [pgd, untargetedf0, 1],
        ]

    def train(self):
        """
        Training configuration.
        """

        writer = SummaryWriter('%s/logs/' % self.directory, max_queue=100)

        epochs = 150
        snapshot = 1

        model_file = '%s/classifier.pth.tar' % self.directory
        incomplete_model_file = find_incomplete_state_file(model_file)
        load_file = model_file
        if incomplete_model_file is not None:
            load_file = incomplete_model_file

        start_epoch = 0
        if os.path.exists(load_file):
            state = common.state.State.load(load_file)
            self.model = state.model
            start_epoch = state.epoch + 1
            # epoch = start_epoch
            log('loaded %s' % load_file)
        else:
            num_classes = 2
            # self.model = models.ResNet(num_classes, [self.trainset.images.shape[3], self.trainset.images.shape[1], self.trainset.images.shape[2]],
            #                            blocks=[3, 3, 3])
            self.model = get_resnet(num_classes)
        self.model = self.model.cuda()

        # optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=10**(-5), betas=(0, 0.99))
        # gamma=0.97
        # batches_per_epoch = len(self.trainloader)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2,
        #                                                    verbose=False, min_lr=1e-6)
        scheduler = None
        attack, objective = self.get_training_attack()
        trainer = AdversarialTraining(self.model, self.trainloader, self.testloader, optimizer, scheduler,
                                                    attack, objective, writer=writer, cuda=True)

        self.model.train()
        for epoch in tqdm(range(start_epoch, epochs)):
            gc.collect()
            trainer.step(epoch)
            writer.flush()

            snapshot_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch)
            common.state.State.checkpoint(snapshot_model_file, self.model, optimizer, scheduler, epoch)

            previous_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch - 1)
            if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
                os.unlink(previous_model_file)

        previous_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch - 1)
        if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
            os.unlink(previous_model_file)

        common.state.State.checkpoint(model_file, self.model, optimizer, scheduler, epoch)

    def evaluate(self):
        """
        Evaluate.
        """

        self.model.eval()
        clean_probabilities = common.test.test(self.model, self.testloader, cuda=True)

        total_adversarial_probabilities = None
        total_adversarial_errors = None

        for attack, objective, attempts in self.get_test_attacks():
            _, adversarial_probabilities, _ = common.test.attack(self.model, self.adversarialloader,
                                                                 attack, objective, attempts=attempts, cuda=True)

            relevant_adversarial_probabilities = numpy.copy(adversarial_probabilities)
            relevant_adversarial_probabilities[
                :,
                numpy.arange(relevant_adversarial_probabilities.shape[1]),
                self.testset.labels[:relevant_adversarial_probabilities.shape[1]],
            ] = 0
            assert len(relevant_adversarial_probabilities.shape) == 3
            adversarial_errors = -numpy.max(relevant_adversarial_probabilities, axis=2)

            total_adversarial_probabilities = common.numpy.concatenate(total_adversarial_probabilities, adversarial_probabilities, axis=0)
            total_adversarial_errors = common.numpy.concatenate(total_adversarial_errors, adversarial_errors, axis=0)

        eval = common.eval.AdversarialEvaluation(clean_probabilities, total_adversarial_probabilities,
                                                             self.testset.labels, validation=0.9, errors=total_adversarial_errors)
        log('test error in %%: %g' % (eval.test_error() * 100))
        log('test error @99%%tpr in %%: %g' % (eval.test_error_at_99tpr() * 100))
        log('robust test error in %%: %g' % (eval.robust_test_error() * 100))
        log('robust test error @99%%tpr in %%: %g' % (eval.robust_test_error_at_99tpr() * 100))

    def main(self):
        """
        Main.
        """

        self.setup()

        model_file = '%s/classifier.pth.tar' % self.directory
        # model_file = 'D:\Projects\Diplom\CCAT\.my_models\pretclassifier.pth.tar'
        if not os.path.exists(model_file):
            self.train()
        else:
            state = common.state.State.load(model_file)
            self.model = state.model
            self.model = self.model.cuda()
        # self.evaluate()


if __name__ == '__main__':
    program = Main()
    program.main()