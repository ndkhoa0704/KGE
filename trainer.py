import torch
from torch.utils.data import DataLoader
from data_loader import KGEDataSet, collate, collate_entity, Collator, TestCollator
from utils import move_to_cuda, save_model, load_model, move_to_cpu, all_axis
import torch.nn.functional as F
from logger_config import logger
import json
import gc
from info_nce import InfoNCE
from dissimilarities import l2_dissimilarity
from functools import partial


@torch.no_grad()
def get_tails(dataset, batch_size=512):
    all_entities = []
    for batch in DataLoader(dataset, batch_size, collate_fn=partial(collate, task='All ent')):
        ents = [i['pos'][:, 2, :] for i in batch]
        ents = torch.cat(ents)
        logger.info('ents shape: {}'.format(ents.shape))
        all_entities.append(ents)
    all_entities = torch.cat(list(all_entities), dim=0)
    ents = torch.unique(all_entities, dim=0)
    ents = move_to_cuda(ents)
    gc.collect()
    return ents


class Trainer:
    def __init__(self, model, args):
        self.args = args
        self.use_cuda = args.use_cuda
        self.model = model(128)
        logger.info(self.model)
        self._setup_training()

        self.loss = torch.nn.BCELoss()

        self.train_dataset = None
        self.test_dataset = None
        self.valid_dataset = None
        self.all_dataset = None

        self.all_ents_embs = None
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.lr
        )
        
        # self.entities_emb = torch.load('entities.pt')

        self.task = args.task


    def test_step(self, DataSet):
        '''
        Evaluate the model on test or valid datasets
        '''
        logger.info('Evaluating')
        collator = TestCollator(self.all_ents_embs)
        self.model.eval()
        logs = []           
        dataloader =  DataLoader(
            dataset=DataSet,
            batch_size=self.args.batch_size,
            collate_fn=partial(collator, task='test')
        )
        
        logs = []

        step = 0
        total_steps = len(dataloader)

        with torch.no_grad():
            for batch in dataloader: 
                
                batch_size = len(batch)
                predicted_scores = []
                all_scores = []
                for data in batch:
                    predicted_tail = self.model(data['pos'][:, 0, :], data['pos'][:, 1, :])
                    predicted_scores.append(self.score_fn(data['pos'][:, 2, :], predicted_tail))
                    all_scores.append(self.score_fn(data['neg'].squeeze(1), predicted_tail))
                    logger.info(data['neg'].squeeze(1).shape)
                    
                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    # logger.info('predicted_tail shape {}'.format(tail[i].shape))
                    argsort = torch.argsort(all_scores[i], dim = 0, descending=False)
                    # logger.info(argsort)
                    # one_tail = move_to_cpu(one_tail)
                    ranking = (argsort == predicted_scores[i]).nonzero()
                    logger.info(ranking)
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
        # return torch.clamp(F.cosine_similarity(true_tail, pred_tail), min=0)
        return F.cosine_similarity(true_tail, pred_tail)
        

    def train_epoch(self, epoch):

        collator = Collator()
        train_data_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=partial(collator, task='train'),
            batch_size=self.args.batch_size
        )
        self.optimizer.zero_grad()
        for batch in train_data_loader:


            pos_losses = None # 32
            neg_losses = None
            target = []

            for data in batch:
                predicted_tail = self.model(data['pos'][:, 0, :], data['pos'][:, 1, :])
                pos_loss = self.score_fn(data['pos'][:, 2, :], predicted_tail) # 1
                # logger.info(pos_loss)
                target += [1] * pos_loss.shape[0]
                neg_loss = self.score_fn(data['neg'].squeeze(1), predicted_tail) # 
                # logger.info(neg_loss)
                target += [0] * neg_loss.shape[0]
                
                if pos_losses is None:
                    pos_losses = pos_loss
                else: 
                    pos_losses = torch.cat((pos_losses, pos_loss))
                if neg_losses is None:
                    neg_losses = neg_loss
                else:
                    neg_losses = torch.cat((neg_losses, neg_loss))


            # # Mean loss
            # loss = (torch.mean(pos_losses) + torch.mean(neg_losses))/2
            
            # BCE loss
            batch_loss = torch.cat((pos_losses, neg_losses))
            loss = self.loss(batch_loss, torch.tensor(target, dtype=torch.float).cuda())

            logger.info('Positive loss: {}'.format(pos_losses.mean()))
            logger.info('Negative loss: {}'.format(neg_losses.mean()))
            logger.info('All loss: {}'.format(loss))

            loss.backward()
            self.optimizer.step()

            logger.info('Epoch: {} - loss: {}'.format(epoch, loss))

        return loss


    def train_loop(self):
        if self.args.task == 'train':
            self.model.train()
            # if self.train_dataset is None:
            self.train_dataset = KGEDataSet(paths=[self.args.train_path])
            best_score = None
            for i in range(self.args.no_epoch):
                score = self.train_epoch(i)
                if best_score is None or best_score < score:
                    save_model(i, self.model, self.optimizer, best=True)
                    best_score = score

            del self.train_dataset
            gc.collect()
            

            # if self.test_dataset =  
            self.all_dataset = KGEDataSet(
                paths=[
                    self.args.train_path,
                    self.args.test_path,
                    self.args.valid_path
                ]
            )

            self.all_ents_embs = get_tails(self.all_dataset)

            
            
            self.test_dataset = KGEDataSet(paths=[self.args.test_path], inverse=False)
            metrics = self.test_step(self.test_dataset)
            
            del self.test_dataset
            gc.collect()

            json.dump(metrics, open('train_metrics.json', 'w'))

        elif self.args.task == 'test':
            self.model.eval()
            self.test_dataset = KGEDataSet(paths=[self.args.test_path], inverse=False)

            # if self.test_dataset =  
            self.all_dataset = KGEDataSet(
                paths=[
                    self.args.train_path,
                    self.args.test_path,
                    self.args.valid_path
                ]
            )

            self.all_ents_embs = get_tails(self.all_dataset)

            logger.info('***Loading checkpoint***')
            self.model, self.optimizer = load_model(epoch=None, model=self.model, optimizer=self.optimizer)
            metrics = self.test_step(self.test_dataset)

            del self.test_dataset
            gc.collect()

            json.dump(metrics, open('test_metrics.json', 'w'))


        elif self.args.task == 'valid':
            self.valid_dataset = KGEDataSet(paths=[self.args.valid_path], inverse=False)
            logger.info('***Loading checkpoint***')
            self.model, self.optimizer = load_model(epoch=None, model=self.model, optimizer=self.optimizer)
            metrics = self.test_step(self.valid_dataset)
            del self.valid_dataset
            gc.collect()

            json.dump(metrics, open('valid_metrics.json', 'w'))
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