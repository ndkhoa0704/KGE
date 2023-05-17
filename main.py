from arguments import args
from trainer import Trainer
from models import CustomTransSmth
from logger_config import logger


if __name__=='__main__':
    try:
        trainer = Trainer(CustomTransSmth, args=args)
        trainer.train_loop()
    except Exception as e:
        logger.critical(e, exc_info=True)
        logger.error(e)
        logger.debug(e)
        logger.exception(e)