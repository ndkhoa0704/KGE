from torch.utils.data import DataLoader
from data_loader import DataSet, collate    
from logger_config import logger
from arguments import args
from trainer import Trainer
from models import CustomTransSmth


if __name__=='__main__':
    trainer = Trainer(CustomTransSmth, args=args)
    trainer.train_loop()