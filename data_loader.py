from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import os
from logger_config import logger
import torch
from copy import deepcopy
from utils import move_to_cuda, move_to_cpu, rowwise_in
from arguments import args as main_args
import numpy as np


BERT_EMBEDDER = None

ENT_DESC = None

    
def get_ent_desc():
    global ENT_DESC
    if ENT_DESC:
        return ENT_DESC
    ENT_DESC = dict()
    with open(main_args.ents_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for d in data:
        ENT_DESC[d['entity_id']] = d['entity_desc']
    return ENT_DESC
    
    
def _parse_entity(entity: str) -> str:
    return entity.replace('_', ' ').replace('\n', ' ').strip()


class Example:
    '''
    Store textual triples
    '''
    def __init__(self, head, relation, tail, head_desc, tail_desc):
        # self.guid = guid
        self.h = head
        self.r = relation
        self.t = tail
        self.h_d = head_desc
        self.t_d = tail_desc

    @property
    def head(self):
        return _parse_entity(self.h)
    
    @property
    def relation(self):
        return _parse_entity(self.r)
    
    @property
    def tail(self):
        return _parse_entity(self.t)
    
    @property
    def head_desc(self):
        return _parse_entity(self.h_d)
    
    @property
    def tail_desc(self):
        return _parse_entity(self.t_d)

    def __str__(self):
        return '''
        head: {}
        relation: {}
        tail: {}
        '''.format(self.h, self.r, self.t)


class BertEmbedder:
    def __init__(self):
        super().__init__()
        pretrained_weights = 'bert-base-uncased'
        self.tokenizer_class = BertTokenizer
        self.model_class = BertModel
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.use_cuda = main_args.use_cuda
        self.hr_model = self.model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        # self.h_model = self.model_class.from_pretrained(pretrained_weights)
        # self.r_model = self.model_class.from_pretrained(pretrained_weights)
        self.t_model = deepcopy(self.hr_model)

        self.hr_model.eval()
        self.t_model.eval()

        if self.use_cuda:
            # self.h_model = self.h_model.to('cuda')
            self.hr_model = self.hr_model.to('cuda')
            # self.h_model = self.h_model.to('cuda')
            # self.r_model = self.r_model.to('cuda')
            self.t_model = self.t_model.to('cuda')


        self.max_word_len = main_args.max_word_len
        self.max_seq_len = 3*main_args.max_word_len # h + r + t


    def get_bert_embeddings(self, examples) -> dict:

        features = convert_examples_to_features(examples, self.tokenizer, max_word_len=self.max_word_len)
        
        h_token_ids = torch.tensor([f['h_token_id'] for f in features], dtype=torch.long)
        r_token_ids = torch.tensor([f['r_token_id'] for f in features], dtype=torch.long)
        t_token_ids = torch.tensor([f['t_token_id'] for f in features], dtype=torch.long)

        if self.use_cuda:
            h_token_ids = move_to_cuda(h_token_ids)
            r_token_ids = move_to_cuda(r_token_ids)
            t_token_ids = move_to_cuda(t_token_ids)

        # print(h_token_ids.shape)
        logger.info('***Geting embedding***')

        # segment_tensor = torch.tensor([[1] * len(h_token_ids)] * len(examples))
        with torch.no_grad():
            # h_last_hidden_states = torch.stack(self.hr_model(h_token_ids)[2])  # Models outputs are now tuples
            # r_last_hidden_states = torch.stack(self.hr_model(r_token_ids)[2])
            # t_last_hidden_states = torch.stack(self.t_model(t_token_ids)[2])


            h_last_hidden_states = self.hr_model(h_token_ids)[2][-2] # Models outputs are now tuples
            r_last_hidden_states = self.hr_model(r_token_ids)[2][-2]
            t_last_hidden_states = self.t_model(t_token_ids)[2][-2]

            h_last_hidden_states = torch.mean(h_last_hidden_states, dim=1)
            r_last_hidden_states = torch.mean(r_last_hidden_states, dim=1)
            t_last_hidden_states = torch.mean(t_last_hidden_states, dim=1)


            h_last_hidden_states = move_to_cpu(h_last_hidden_states)
            r_last_hidden_states = move_to_cpu(r_last_hidden_states)
            t_last_hidden_states = move_to_cpu(t_last_hidden_states)

            

        # logger.info('h emb shape {}'.format(h_last_hidden_states.shape))
        # logger.info('r emb shape {}'.format(r_last_hidden_states.shape))
        # logger.info('t emb shape {}'.format(t_last_hidden_states.shape))
        # exit()


        # if self.use_cuda:
        #     move_to_cpu(h_token_ids)
        #     move_to_cpu(r_token_ids)
        #     move_to_cpu(t_token_ids)

        
        # embeddings = {
        #     'head': torch.mean(hr_last_hidden_states[:, :main_args.max_word_len, :], dim=1),
        #     'relation': torch.mean(hr_last_hidden_states[:, main_args.max_word_len:2*main_args.max_word_len, :], dim=1),
        #     'tail': torch.mean(t_last_hidden_states[:, -main_args.max_word_len:, :], dim=1)
        # }

        # embeddings = {
        #     'head': h_last_hidden_states,
        #     'relation': r_last_hidden_states,
        #     'tail': t_last_hidden_states
        # }

        # logger.info('h emb shape {}'.format(embeddings['head'].shape))
        # logger.info('r emb shape {}'.format(embeddings['relation'].shape))
        # logger.info('t emb shape {}'.format(embeddings['tail'].shape))

        embeddings = move_to_cuda(torch.stack((h_last_hidden_states, r_last_hidden_states, t_last_hidden_states), dim=1))

        # logger.info('batch size: {}'.format())
        return embeddings
    
    def get_ent_emb(self, examples) -> dict:

        features = convert_examples_to_features(examples, self.tokenizer, max_word_len=self.max_word_len)
        t_token_ids = torch.tensor(list(set(tuple([f['t_token_id'] for f in features]))), dtype=torch.long)
        print(t_token_ids.shape)
        exit()

        if self.use_cuda:
            t_token_ids = move_to_cuda(t_token_ids)

        # print(h_token_ids.shape)
        logger.info('***Geting embedding***')

        # segment_tensor = torch.tensor([[1] * len(h_token_ids)] * len(examples))
        with torch.no_grad():
            t_last_hidden_states = self.t_model(t_token_ids)[2][-2]
            t_last_hidden_states = torch.mean(t_last_hidden_states, dim=1)
            t_last_hidden_states = move_to_cpu(t_last_hidden_states)

        embeddings = move_to_cuda(t_last_hidden_states)

        # logger.info('batch size: {}'.format())
        return embeddings

    

def get_bert_embedder():
    global BERT_EMBEDDER
    if BERT_EMBEDDER is None:
        BERT_EMBEDDER = BertEmbedder()
    return BERT_EMBEDDER


def _truncate_and_padding_embedding(word: list, max_len) -> torch.tensor:
    '''Truncate or add padding to each entity/relation embedding'''
    # print(word)
    word = ['[CLS]'] + word
    if len(word) > max_len - 1:
        return word[:max_len - 1] + ['[SEP]']
    elif len(word) < max_len - 1:
        return word + ['[PAD]'] * (max_len - len(word) - 1) + ['[SEP]']
    word += ['[SEP]']
    # print(word)
    return word


def convert_examples_to_features(
        examples, 
        tokenizer: BertTokenizer, 
        max_word_len=7
):
    '''
    Covert text triplets to tokenized features
    
    :return: tokenized features (hr, t) (h, rt)
    :rtype tuple
    '''
    features = []
    for i, example in enumerate(examples):
        if i % 10000 == 0:
            logger.info("Writing example %d of %d" % (i, len(examples)))
        token_h = tokenizer.tokenize(example.head) + tokenizer.tokenize(example.head_desc)
        token_r = tokenizer.tokenize(example.relation)
        token_t = tokenizer.tokenize(example.tail) + tokenizer.tokenize(example.tail_desc)
        # token_t = tokenizer.tokenize(example.tail)


        token_h = _truncate_and_padding_embedding(token_h, max_len=max_word_len)
        token_r = _truncate_and_padding_embedding(token_r, max_len=max_word_len)
        token_t = _truncate_and_padding_embedding(token_t, max_len=max_word_len)

        tokenized_h = tuple(tokenizer.convert_tokens_to_ids(token_h))
        tokenized_r = tuple(tokenizer.convert_tokens_to_ids(token_r))
        tokenized_t = tuple(tokenizer.convert_tokens_to_ids(token_t))


        features.append({
            'h_token_id': tokenized_h,
            'r_token_id': tokenized_r,
            't_token_id': tokenized_t
            # 'hr_token_id': tokens[:2*max_word_len],
            # 'rt_token_id': tokens[max_word_len:],
        })

        examples[i] = None

    return features


def load_data(path, inverse=True):
    '''Load data and create examples'''
    logger.info('Data path: {}'.format(path))

    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info('Load {} triplets from {}'.format(len(data), path))

    examples = []
    ent_descr = get_ent_desc()
    for i in range(len(data)):
        examples.append(Example(
            # guid=f'{task}-{i}',
            head=data[i]['head'],
            relation=data[i]['relation'],
            tail=data[i]['tail'],
            head_desc=ent_descr[data[i]['head_id']],
            tail_desc=ent_descr[data[i]['tail_id']]
        ))
        if inverse:
            examples.append(Example(
                # guid=f'{task}-{i}-inv',
                head=data[i]['tail'],
                relation='inverse ' + data[i]['relation'],
                tail=data[i]['head'],
                head_desc=ent_descr[data[i]['tail_id']],
                tail_desc=ent_descr[data[i]['head_id']]
            ))
        data[i] = None
    return examples


def collate(batch_data, pb=None, **kwargs) -> list:
    embedder = get_bert_embedder()
    neg_samp = None
    embs = embedder.get_bert_embeddings(batch_data, *kwargs)
    pb = torch.rand((512, 3, 768))
    for triple in embs:
        # Self negatives
        if neg_samp is None:
            neg_samp = torch.unsqueeze(torch.cat((torch.unsqueeze(triple[0, :], 0), triple[:-1,:]), dim=0), 1)
        else: 
            neg_samp = torch.cat((
                neg_samp, 
                torch.unsqueeze(torch.cat((torch.unsqueeze(triple[0, :], 0), triple[:-1,:]), dim=0), 1)
            ))
        # Pre-batch
        if pb is not None: # 
            print(triple[:-1,:].shape)
            tmp = torch.cat((triple[:-1,:].expand((1024, *triple[:-1,:].shape)), \
                             pb[:, 2, :][rowwise_in(pb[:, 2, :], torch.unsqueeze(triple[2, :], 0))].unsqueeze(1)), dim=0)
            print(tmp.shape)
            neg_samp = torch.cat((neg_samp, tmp))
        # In-batch
        tmp = torch.cat((triple[:-1,:].expand((1024, *triple[:-1,:].shape)), \
                            embs[:, 2, :][rowwise_in(embs[:, 2, :], torch.unsqueeze(triple[2, :], 0))].unsqueeze(1)), dim=0)
    return embs, neg_samp



def collate_entity(batch_data, **kwargs) -> list:
    embedder = get_bert_embedder()
    return embedder.get_ent_emb(batch_data, *kwargs)


class KGEDataSet(Dataset):
    def __init__(self, paths: list[str], *args, **kwargs):
        self.examples = []
        for path in paths:
            self.examples += load_data(path, *args, **kwargs)
        self.data_len = len(self.examples)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.examples[index]
    

class EntitiesDataset(Dataset):
    def __init__(self, entities):
        self.ents = entities
        self.data_len = len(entities)

    def __getitem__(self, index):
        return self.ents[index]
    
    def __len__(self):
        return self.data_len
        

def get_all_entities(train_path, test_path, valid_path):
    all_dataset = KGEDataSet(
        train_path=train_path,
        test_path=test_path,
        valid_path=valid_path
    )

    all_entities = []
    for batch in DataLoader(all_dataset, batch_size=main_args.batch_size):
        # all_entities.append(batch[:, 0, :])
        all_entities.append(batch[:, 2, :])

    all_entities = torch.cat(list(all_entities), dim=0)
    # print(all_entities[:5])
    ents = torch.unique(all_entities, dim=0)
    # exit()
    print(ents.shape)
    # exit()
    ents = move_to_cpu(ents)
    return ents