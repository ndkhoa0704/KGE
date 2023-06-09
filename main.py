from arguments import args
from trainer import Trainer
from models import CustomTransSmth
from logger_config import logger


if __name__=='__main__':
    if args.task == 'train':
        log_file = 'train.log'
    elif args.task == 'test':
        log_file = 'test.log'
    elif args.task == 'eval':
        log_file = 'eval.log'
        
    with open(log_file, 'w'):
        pass
    try:
        trainer = Trainer(CustomTransSmth, args=args)
        trainer.train_loop()
    except Exception as e:
        logger.critical(e, exc_info=True)
        logger.error(e)
        logger.debug(e)
        logger.exception(e)