from transformers import BertModel, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import json
import os
from logger_config import logger
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import numpy as np


BERT_EMBEDDER = None

def get_bert_embedder(*args, **kwargs):
    global BERT_EMBEDDER
    if BERT_EMBEDDER is None:
        BERT_EMBEDDER = BertEmbedder(*args, **kwargs)
    return BERT_EMBEDDER


class Example:
    '''
    Store textual triples
    '''
    def __init__(self, guid, head, relation, tail):
        self.guid = guid
        self.head = head
        self.relation = relation
        self.tail = tail

class Feature:
    '''Store Bert features'''
    def __init__(self, token_ids, segment_ids, attention_mask, token_indices=None):
        self.token_ids = token_ids
        self.segment_ids = segment_ids
        self.attention_mask = attention_mask
        self.token_indices: dict(tuple)= token_indices # [token_a, token_b, token_cc]

class TripleEmbeddings:
    '''
    Store triple embeddings
    '''
    def __init__(self, head, relation, tail):
        self.head = head
        self.relation = relation
        self.tail = tail

    def __str__(self) -> str:
        return '''
        head: {}
        relation: {}
        tail: {}
        '''.format(self.head, self.relation, self.tail)


def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()


def convert_examples_to_features(examples: list[Example], tokenizer: BertTokenizerFast, max_seq_len=20) -> list[Example]:
    '''Covert Example to '''
    features = []
    for i, example in enumerate(examples):
        if i % 10000 == 0:
            logger.info("Writing example %d of %d" % (i, len(examples)))
        token_a = tokenizer.tokenize(example.head)
        token_b = tokenizer.tokenize(example.relation)
        token_c = tokenizer.tokenize(example.tail)

        _truncate_seq_triple(token_a, token_b, token_c, max_length=max_seq_len - 1)

        token_seq = token_a + token_b + token_c + ['[SEP]']

        token_ids = tokenizer.convert_tokens_to_ids(token_seq)
        attention_mask = [1] * len(token_seq)
        segment_ids = [1] * len(token_seq)

        padding_length = max_seq_len - len(token_ids)

        token_ids += [0] * padding_length
        attention_mask += [0] * padding_length
        segment_ids += [0] * padding_length


        token_a_idx = (0, len(token_a))
        token_b_idx = (len(token_a), len(token_a) + len(token_b))
        token_c_idx = (len(token_a) + len(token_b), len(token_a) + len(token_b) + len(token_c))

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in token_seq]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(Feature(
            token_ids=token_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            token_indices={
                'head': token_a_idx, 
                'relation': token_b_idx, 
                'tail': token_c_idx
            }
        ))

        examples[i] = None

    return features


class BertEmbedder:
    def __init__(self,
                 pretrained_weights='bert-base-uncased',
                 tokenizer_class=BertTokenizerFast,
                 model_class=BertModel,
                 max_seq_len=20):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.model = self.model_class.from_pretrained(pretrained_weights)
        self.max_seq_len = max_seq_len
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)

    def get_bert_embeddings(self, examples: list[Example]) -> torch.tensor:
        features = convert_examples_to_features(examples, self.tokenizer)

        all_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        logger.info('***Geting embedding***')
        last_hidden_states = self.model(all_token_ids)[0]  # Models outputs are now tuples
        embeddings = [TripleEmbeddings(
            head=last_hidden_states[
                features[i].token_indices['head'][0]:features[i].token_indices['head'][1]
            ],
            relation=last_hidden_states[
                features[i].token_indices['relation'][0]:features[i].token_indices['relation'][1]
            ],
            tail=last_hidden_states[
                features[i].token_indices['tail'][0]:features[i].token_indices['tail'][1]
            ],
        ) for i in range(len(examples))]
        return embeddings


def load_data(path, task, inverse=True):
    '''Load data and create examples'''
    assert task and task in ['train', 'test', 'valid']
    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info('Load {} triplets from {}'.format(len(data), path))

    examples = []
    for i in range(len(data)):
        examples.append(Example(
            guid=f'{task}-{i}',
            head=data[i]['head'],
            relation=data[i]['relation'],
            tail=data[i]['tail']
        ))
        if inverse:
            examples.append(Example(
                guid=f'{task}-{i}-inv',
                head=data[i]['tail'],
                relation=data[i]['relation'] + '_inverse',
                tail=data[i]['head']  
            ))
        data[i] = None

    return examples


def collate(batch_data: list[Example]) -> list:
    embedder = get_bert_embedder()
    return embedder.get_bert_embeddings(batch_data)


class DataSet(Dataset):
    def __init__(self, path, task, *args, **kwargs):
        self.examples = load_data(path, task, *args, **kwargs)
        self.data_len = len(self.examples)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.examples[index]
    

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    # @staticmethod
    # def test_step(model, test_triples, all_true_triples, args):
    #     '''
    #     Evaluate the model on test or valid datasets
    #     '''
        
    #     model.eval()
        
    #     if args.countries:
    #         #Countries S* datasets are evaluated on AUC-PR
    #         #Process test data for AUC-PR evaluation
    #         sample = list()
    #         y_true  = list()
    #         for head, relation, tail in test_triples:
    #             for candidate_region in args.regions:
    #                 y_true.append(1 if candidate_region == tail else 0)
    #                 sample.append((head, relation, candidate_region))

    #         sample = torch.LongTensor(sample)
    #         if args.cuda:
    #             sample = sample.cuda()

    #         with torch.no_grad():
    #             y_score = model(sample).squeeze(1).cpu().numpy()

    #         y_true = np.array(y_true)

    #         #average_precision_score is the same as auc_pr
    #         auc_pr = average_precision_score(y_true, y_score)

    #         metrics = {'auc_pr': auc_pr}
            
        # else:
        #     #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #     #Prepare dataloader for evaluation
        #     test_dataloader_head = DataLoader(
        #         TestDataset(
        #             test_triples, 
        #             all_true_triples, 
        #             args.nentity, 
        #             args.nrelation, 
        #             'head-batch'
        #         ), 
        #         batch_size=args.test_batch_size,
        #         num_workers=max(1, args.cpu_num//2), 
        #         collate_fn=TestDataset.collate_fn
        #     )

        #     test_dataloader_tail = DataLoader(
        #         TestDataset(
        #             test_triples, 
        #             all_true_triples, 
        #             args.nentity, 
        #             args.nrelation, 
        #             'tail-batch'
        #         ), 
        #         batch_size=args.test_batch_size,
        #         num_workers=max(1, args.cpu_num//2), 
        #         collate_fn=TestDataset.collate_fn
        #     )
            
        #     test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
        #     logs = []

        #     step = 0
        #     total_steps = sum([len(dataset) for dataset in test_dataset_list])

        #     with torch.no_grad():
        #         for test_dataset in test_dataset_list:
        #             for positive_sample, negative_sample, filter_bias, mode in test_dataset:
        #                 if args.cuda:
        #                     positive_sample = positive_sample.cuda()
        #                     negative_sample = negative_sample.cuda()
        #                     filter_bias = filter_bias.cuda()

        #                 batch_size = positive_sample.size(0)

        #                 score = model((positive_sample, negative_sample), mode)
        #                 score += filter_bias

        #                 #Explicitly sort all the entities to ensure that there is no test exposure bias
        #                 argsort = torch.argsort(score, dim = 1, descending=True)

        #                 if mode == 'head-batch':
        #                     positive_arg = positive_sample[:, 0]
        #                 elif mode == 'tail-batch':
        #                     positive_arg = positive_sample[:, 2]
        #                 else:
        #                     raise ValueError('mode %s not supported' % mode)

        #                 for i in range(batch_size):
        #                     #Notice that argsort is not ranking
        #                     ranking = (argsort[i, :] == positive_arg[i]).nonzero()
        #                     assert ranking.size(0) == 1

        #                     #ranking + 1 is the true ranking used in evaluation metrics
        #                     ranking = 1 + ranking.item()
        #                     logs.append({
        #                         'MRR': 1.0/ranking,
        #                         'MR': float(ranking),
        #                         'HITS@1': 1.0 if ranking <= 1 else 0.0,
        #                         'HITS@3': 1.0 if ranking <= 3 else 0.0,
        #                         'HITS@10': 1.0 if ranking <= 10 else 0.0,
        #                     })

        #                 if step % args.test_log_steps == 0:
        #                     logger.info('Evaluating the model... (%d/%d)' % (step, total_steps))

        #                 step += 1

        #     metrics = {}
        #     for metric in logs[0].keys():
        #         metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        # return metrics


if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--data-path', dest='data_path')
    parser.add_argument('--batch-size', dest='batch_size', default=1024)

    args = parser.parse_args()

    train_data = DataLoader(
        DataSet(args.data_path, task='train'),
        # num_workers=4,
        collate_fn=collate,
        batch_size=args.batch_size,
        shuffle=True,
    )

    for batch in train_data:
        print(batch[0])
        break