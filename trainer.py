import torch
from torch.utils.data import DataLoader
from data_loader import DataSet, collate
from utils import move_to_cuda
import torch.nn.functional as F
from logger_config import logger
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(self.args.gamma, 768)
        logger.info(self.model)
        self._setup_training()
        self.train_dataset = DataSet(self.args.train_path, self.args.task)
        self.train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate,
            batch_size=args.batch_size
        )

        current_learning_rate = self.args.learning_rate
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=current_learning_rate
        )

    def train_epoch(self, epoch):
        for batch in self.train_data_loader:
            # self.train_sampler.set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad()
            
            if self.args.use_cuda and torch.cuda.is_available():
                batch = move_to_cuda(batch)

            score = self.model(batch)
            pos_score = F.logsigmoid(score).squeeze(dim = 1)

            loss = - pos_score.mean()
            loss.backward()

            self.optimizer.step()
            logger.info('Epoch: {} - loss: {}'.format(epoch, loss))


    def train_loop(self):
        for i in range(self.args.no_epoch):
            self.train_epoch(i)


    def _setup_training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')