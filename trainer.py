import torch
from torch.utils.data import DataLoader
from data_loader import KGEDataSet
from utils import move_to_cuda, save_model, load_model, move_to_cpu, HashTensorWrapper
import torch.nn.functional as F
from logger_config import logger
# from dissimilarities import l1_dissimilarity, l2_dissimilarity
from functools import partial
from prepare_ent import get_entities
from data_loader import get_bert_embedder


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(self.args.gamma, 768)
        logger.info(self.model)
        self._setup_training()
        self.train_dataset = None
        self.test_dataset = None
        self.all_dataset = None
        self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )
        self.entities_emb = None
        # self.entities_emb = torch.load('entities.pt')

        self.task = args.task

    def _get_all_entities(self):
        if self.all_dataset is None:
            self.all_dataset = KGEDataSet(
                train_path=self.args.train_path,
                test_path=self.args.test_path,
                valid_path=self.args.valid_path
            )

        all_entities = []
        for batch in DataLoader(self.all_dataset):
            # all_entities.append(batch[:, 0, :])
            all_entities.append(batch[:, 2, :])

        all_entities = torch.cat(list(all_entities), dim=0)
        print(all_entities.shape)
        # print(all_entities[:5])
        ents = torch.unique(all_entities, dim=0)
        print(ents.shape)
        # exit()
        return move_to_cpu(ents)



    def test_step(self):
        '''
        Evaluate the model on test or valid datasets
        '''
        logger.info('Evaluating')
        
        self.model.eval()
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
        
        logs = []           
        test_dataloader =  DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=self.args.batch_size
        )
        
        logs = []

        step = 0
        total_steps = len(test_dataloader)

        def all_axis(tensor):
            new_tensor = []
            for i in tensor:
                new_tensor.append(torch.all(i))
            new_tensor = torch.tensor(new_tensor)
            return new_tensor


        with torch.no_grad():
            for batch in test_dataloader: 
                if self.args.use_cuda and torch.cuda.is_available():
                    batch = move_to_cuda(batch)

                batch_size = batch.size(0)

                tail = self.model(batch[:, 0, :], batch[:, 1, :])
                if self.args.use_cuda:
                    tail = move_to_cpu(tail)

                for i in range(batch_size):
                    #Notice that argsort is not ranking
                    score = self.score_fn(tail[i], self.entities_emb)
                    # flatten_tail = tail[i].flatten(start_dim=1, end_dim=2)
                    # flatten_ents = self.entities_emb.flatten(start_dim=1, end_dim=2)
                    # score = torch.abs(self.sim(flatten_tail, flatten_ents) - 1) 
                    argsort = torch.argsort(score, dim = 0, descending=False)
                    ents = torch.index_select(self.entities_emb,0, argsort)
                    one_tail = move_to_cpu(batch[i, 2, :])
                    ranking = all_axis(ents == one_tail).nonzero()
                    if ranking.shape[0] == 0:
                        continue
                    # print(ranking.shape)
                    # assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    met = {
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    }
                    logger.info(met)
                    logs.append(met)

                if step % self.args.test_log_steps == 0:
                    logger.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)
        
        logger.info(metrics)

        return metrics

    def score_fn(self, true_tail, pred_tail):
        score = torch.abs(pred_tail - true_tail)
        score = torch.norm(score, p=2, dim=1)
        # logger.info('score shape {}'.format(score.shape))
        score = F.sigmoid(score)
        # return score
        return score
        # return -self.sim(pred_tail, true_tail)

    # def score_fn(self, true_tail, pred_tail):
    #     score = torch.sum(true_tail * pred_tail, dim=1)
    #     # logger.info('score shape {}'.format(score.shape))
    #     score = F.relu(score)
    #     # return score
    #     return score
    #     # return -self.sim(pred_tail, true_tail)


    def train_epoch(self, epoch):
        train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size
        )

        self.model.train()
        self.optimizer.zero_grad()
        for batch in train_data_loader:
            if self.args.use_cuda and torch.cuda.is_available():
                batch = move_to_cuda(batch)
            
            logger.info('Batch size {}'.format(batch.shape))
            tail = self.model(batch[:, 0, :], batch[:, 1, :])

            score = self.score_fn(batch[:, 2, :], tail)

            loss = - score.mean()

            loss.backward()
            self.optimizer.step()

            logger.info('Epoch: {} - loss: {}'.format(epoch, loss))


    def train_loop(self):
        if self.args.task == 'train':
            if self.train_dataset is None:
                self.train_dataset = KGEDataSet(train_path=self.args.train_path)
            if self.test_dataset is None:
                self.test_dataset = KGEDataSet(test_path=self.args.test_path)
            if self.entities_emb is None:
                self.entities_emb = self._get_all_entities()
            for i in range(self.args.no_epoch):
                self.train_epoch(i)
                save_model(i, self.model, self.optimizer)
            self.test_step()
        elif self.args.task == 'test':
            if self.test_dataset is None:
                self.test_dataset = KGEDataSet(test_path=self.args.test_path)
            if self.entities_emb is None:
                self.entities_emb = self._get_all_entities()
            logger.info('***Loading checkpoint***')
            self.model, self.optimizer = load_model(epoch=20, model=self.model, optimizer=self.optimizer)
            self.test_step()
        elif self.args.task == 'valid':
            # TODO
            raise NotImplementedError()
        else:
            raise Exception('Unsupported task')


    def _setup_training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.cuda()
        else:
            logger.info('No gpu will be used')