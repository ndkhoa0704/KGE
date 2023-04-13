import torch
from torch.utils.data import DataLoader
from data_loader import TrainDataSet, TestDataSet, collate, get_entities_emb
from utils import move_to_cuda, save_model
import torch.nn.functional as F
from logger_config import logger
from dissimilarities import l1_dissimilarity, l2_dissimilarity
from functools import partial


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(self.args.gamma, 768)
        logger.info(self.model)
        self._setup_training()
        self.train_dataset = TrainDataSet(self.args.train_path, 'train')
        self.test_dataset = TestDataSet(
            train_path=self.args.train_path,
            test_path=self.args.test_path,
            valid_path=self.args.valid_path,
            task='test'
        )

        self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )
        self.entitiies_embs = get_entities_emb(self.test_dataset)


    def test_step(self):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        self.model.eval()
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
        
        logs = []           
        test_dataloader_head =  DataLoader(
            self.test_dataset,
            shuffle=True,
            collate_fn=lambda batch: collate(batch, mode='forward'),
            batch_size=self.args.batch_size
        )

        test_dataloader_tail =  DataLoader(
            self.test_dataset,
            shuffle=True,
            collate_fn=lambda batch: collate(batch, mode='backward'),
            batch_size=self.args.batch_size
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        step = 0
        total_steps = len(self.test_data_loader)

        with torch.no_grad():
            for batch in self.test_data_loader: 
                if self.args.use_cuda and torch.cuda.is_available():
                    batch = move_to_cuda(batch)

                batch_size = batch['head'].size(0)

                tail = self.model(batch['head'], batch['relation'])
                # argsort = torch.argsort(score, dim = 1, descending=True)

                # score += filter_bias

                #Explicitly sort all the entities to ensure that there is no test exposure bias
                # argsort = torch.argsort(score, dim = 1, descending=True)

                # if mode == 'head-batch':
                #     positive_arg = positive_sample[:, 0]
                # elif mode == 'tail-batch':
                #     positive_arg = positive_sample[:, 2]
                # else:
                #     raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    score = self.score_fn(tail[i], self.entitiies_embs)
                    argsort = self.entitiies_embs[torch.argsort(score, dim = 0, descending=False)]
                    ranking = (argsort == batch['tail'][i]).nonzero()

                    # assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % self.args.test_log_steps == 0:
                    logger.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics

    def score_fn(self, true_tail, pred_tail):
        # score = torch.norm(score, p=1, dim=1)
        # return F.logsigmoid(score).squeeze(dim = 0)
        return l1_dissimilarity(pred_tail, true_tail)


    def train_epoch(self, epoch):
        train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=lambda batch: collate(batch, mode='all'),
            batch_size=self.args.batch_size
        )
        for batch in train_data_loader:
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

        # self.test_step()


    def _setup_training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')