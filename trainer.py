import torch
from torch.utils.data import DataLoader
from data_loader import KGEDataSet, EntitiesDataset
from utils import move_to_cuda, save_model, load_model, move_to_cpu, all_axis
import torch.nn.functional as F
from logger_config import logger
from functools import partial
from prepare_ent import get_entities
from data_loader import get_bert_embedder, get_all_entities
from dissimilarities import l2_torus_dissimilarity
import json


# def get_ents(dataset):
#     all_entities = []
#     for batch in DataLoader(dataset):
#         all_entities.append(batch[:, 2, :])

#     all_entities = torch.cat(list(all_entities), dim=0)
#     ents = torch.unique(all_entities, dim=0)
#     return move_to_cpu(ents)


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(768)
        logger.info(self.model)
        self._setup_training()
        # self.train_dataset = KGEDataSet(train_path=self.args.train_path)
        self.test_dataset = KGEDataSet(test_path=self.args.test_path)
        self.valid_dataset = KGEDataSet(valid_path=self.args.valid_path)
        self.all_ents_embs = get_all_entities(
            train_path=self.args.train_path,
            test_path=self.args.test_path,
            valid_path=self.args.valid_path
        )
        # self.all_ents_embs = None
        # self.all_dataset = None
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )
        self.entities_dataset = None
        # self.entities_emb = torch.load('entities.pt')

        self.task = args.task


    def test_step(self, DataSet):
        '''
        Evaluate the model on test or valid datasets
        '''
        logger.info('Evaluating')
        
        self.model.eval()
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
        
        logs = []           
        dataloader =  DataLoader(
            DataSet,
            batch_size=self.args.batch_size
        )

        ent_dataloader = DataLoader(
            dataset=EntitiesDataset(self.all_ents_embs),
            batch_size=self.args.batch_size
        )
        
        logs = []

        step = 0
        total_steps = len(dataloader)

        with torch.no_grad():
            for batch in dataloader: 
                if self.args.use_cuda and torch.cuda.is_available():
                    batch = move_to_cuda(batch)

                batch_size = batch.size(0)
                tail = self.model(batch[:, 0, :], batch[:, 1, :])


                # if self.args.use_cuda:
                #     tail = move_to_cpu(tail)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    # logger.info('predicted_tail shape {}'.format(tail[i].shape))
                    score = None
                    one_tail = tail[i]
                    for ent_batch in ent_dataloader:
                        batch_score = self.test_batch(ent_batch, one_tail.reshape(1, *tail[i].shape))
                        if score is None:
                            score = batch_score
                        else:
                            score = torch.cat((score, batch_score))
                    
                    # score = self.score_fn(tail[i].reshape(1, *tail[i].shape), all_ents)
                    argsort = torch.argsort(score, dim = 0, descending=False)
                    ents = torch.index_select(self.all_ents_embs,0, argsort)
                    # one_tail = move_to_cpu(batch[i, 2, :])
                    ranking = all_axis(ents == one_tail).nonzero()
                    # print(ranking.shape)
                    assert ranking.size(0) == 1

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

        # return l2_torus_dissimilarity(true_tail, pred_tail)
        logger.debug('true tail shape: {}'.format(true_tail.shape))
        logger.debug('pred tail shape: {}'.format(pred_tail.shape))
        return F.cosine_similarity(pred_tail.flatten(1,2), true_tail.flatten(1,2))
    


    # def score_fn(self, true_tail, pred_tail):
    #     score = torch.sum(true_tail * pred_tail, dim=1)
    #     # logger.info('score shape {}'.format(score.shape))
    #     score = F.relu(score)
    #     # return score
    #     return score
    #     # return -self.sim(pred_tail, true_tail)


    def test_batch(self, batch, pred_tail):
        if self.args.use_cuda:
            batch = move_to_cuda(batch)
            tail = move_to_cuda(pred_tail)
            score = self.score_fn(tail, batch[:,2,:])
            batch = move_to_cpu(batch)
            tail = move_to_cpu(tail)
        else:
            score = self.score_fn(pred_tail, batch[:,2,:])
        return score
        
        

    def train_epoch(self, epoch):
        train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size
        )

        self.model.train()
        self.optimizer.zero_grad()
        scores = []
        for batch in train_data_loader:
            if self.args.use_cuda and torch.cuda.is_available():
                batch = move_to_cuda(batch)
            
            logger.info('Batch size {}'.format(batch.shape))
            tail = self.model(batch[:, 0, :], batch[:, 1, :])

            score = self.score_fn(batch[:, 2, :], tail)

            loss = score.mean()
            scores.append(loss)

            loss.backward()
            self.optimizer.step()

            logger.info('Epoch: {} - loss: {}'.format(epoch, loss))
        return torch.tensor(scores).mean()


    def train_loop(self):
        if self.args.task == 'train':
            best_score = None    
            for i in range(self.args.no_epoch):
                score = self.train_epoch(i)
                if best_score is None or best_score > score:
                    save_model(i, self.model, self.optimizer, best=True)
                    best_score = score
                    metrics = self.test_step(self.test_dataset)
            json.dump(metrics, open('metrics.json', 'w'))
        elif self.args.task == 'test':
            logger.info('***Loading checkpoint***')
            self.model, self.optimizer = load_model(epoch=None, model=self.model, optimizer=self.optimizer)
            metrics = self.test_step(self.valid_dataset)
            json.dump(metrics, open('metrics.json', 'w'))
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