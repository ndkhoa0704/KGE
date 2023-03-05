import torch
from torch.utils.data import DataLoader
from data_loader import DataSet, collate
from utils import move_to_cuda, save_model
import torch.nn.functional as F
from logger_config import logger


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(self.args.gamma, 768)
        logger.info(self.model)
        self._setup_training()
        self.train_dataset = DataSet(self.args.train_path, 'train')
        self.test_dataset = DataSet(self.args.test_path, 'test')
        self.train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate,
            batch_size=args.batch_size
        )
        self.test_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate,
            batch_size=args.batch_size
        )
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )


    def test_step(self, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        self.model.eval()
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
        
        logs = []

        step = 0
        total_steps = len(self.test_data_loader)

        with torch.no_grad():
            for batch in self.test_data_loader:
                if args.cuda:
                    batch = move_to_cuda(batch)

                batch_size = batch.size(0)

                tail = self.model(batch['head'], batch['relation'])
                score = self.score_fn(tail, batch['tail'])
                # score += filter_bias

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim = 1, descending=True)

                # if mode == 'head-batch':
                #     positive_arg = positive_sample[:, 0]
                # elif mode == 'tail-batch':
                #     positive_arg = positive_sample[:, 2]
                # else:
                #     raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == batch[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logger.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

    def score_fn(self, true_tail, pred_tail):
        return F.logsigmoid(pred_tail - true_tail).squeeze(dim = 0)


    def train_epoch(self, epoch):
        for batch in self.train_data_loader:
            # self.train_sampler.set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad()
            
            if self.args.use_cuda and torch.cuda.is_available():
                batch = move_to_cuda(batch)

            tail = self.model(batch['head'], batch['relation'])

            score = self.score_fn(batch['tail'], tail)

            loss = - score.mean()
            loss.backward()

            self.optimizer.step()
            logger.info('Epoch: {} - loss: {}'.format(epoch, loss))


    def train_loop(self):
        for i in range(self.args.no_epoch):
            self.train_epoch(i)
            save_model(i, self.model, self.optimizer)

        self.test_step()


    def _setup_training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')