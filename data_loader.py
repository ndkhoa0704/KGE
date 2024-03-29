from typing import Any
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
from dataclasses import dataclass


BERT_EMBEDDER = None

ENT_DESC = None

ALL_ENT_IDS = None

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
    def __init__(self, head, head_id, relation, tail, tail_id, head_desc, tail_desc):
        # self.guid = guid
        self.h = head
        self.r = relation
        self.t = tail
        self.h_d = head_desc
        self.t_d = tail_desc
        self.h_id = head_id
        self.t_id = tail_id


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
        pretrained_weights = "bert-base-uncased"
        self.tokenizer_class = BertTokenizer
        self.model_class = BertModel
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.use_cuda = main_args.use_cuda
        self.hr_model = self.model_class.from_pretrained(pretrained_weights, output_hidden_states=True)
        self.t_model = deepcopy(self.hr_model)

        self.hr_model.eval()
        self.t_model.eval()

        if self.use_cuda:
            self.hr_model = self.hr_model.to('cuda')
            self.t_model = self.t_model.to('cuda')


        self.max_word_len = main_args.max_word_len
        self.max_seq_len = 3*main_args.max_word_len # h + r + t


    def get_bert_embeddings(self, examples, task=None) -> dict:
        features = convert_examples_to_features(examples, self.tokenizer, max_word_len=self.max_word_len)
        
        h_token_ids = torch.tensor([f['h_token_id'] for f in features], dtype=torch.long)
        r_token_ids = torch.tensor([f['r_token_id'] for f in features], dtype=torch.long)
        t_token_ids = torch.tensor([f['t_token_id'] for f in features], dtype=torch.long)
        h_ids = [f['h_id'] for f in features]
        t_ids = [f['t_id'] for f in features]

        if self.use_cuda:
            h_token_ids = move_to_cuda(h_token_ids)
            r_token_ids = move_to_cuda(r_token_ids)
            t_token_ids = move_to_cuda(t_token_ids)

        # print(h_token_ids.shape)
        logger.info('***Getting embedding for {}***'.format(task))

        # segment_tensor = torch.tensor([[1] * len(h_token_ids)] * len(examples))
        with torch.no_grad():
            h_last_hidden_states = torch.stack(self.hr_model(h_token_ids)[2])  # Models outputs are now tuples
            r_last_hidden_states = torch.stack(self.hr_model(r_token_ids)[2])
            t_last_hidden_states = torch.stack(self.t_model(t_token_ids)[2])


            h_last_hidden_states = self.hr_model(h_token_ids)[2][-2] # Models outputs are now tuples
            r_last_hidden_states = self.hr_model(r_token_ids)[2][-2]
            t_last_hidden_states = self.t_model(t_token_ids)[2][-2]

            # h_last_hidden_states = self.hr_model(h_token_ids)[2] # Models outputs are now tuples
            # r_last_hidden_states = self.hr_model(r_   token_ids)[2]
            # t_last_hidden_states = self.t_model(t_token_ids)[2]

        # h_last_hidden_states = [h_last_hidden_states[i] for i in (-1, -2, -3, -4)]
        # r_last_hidden_states = [r_last_hidden_states[i] for i in (-1, -2, -3, -4)]
        # t_last_hidden_states = [t_last_hidden_states[i] for i in (-1, -2, -3, -4)]

        # h_last_hidden_states = torch.cat(tuple(h_last_hidden_states), dim=-1)
        # r_last_hidden_states = torch.cat(tuple(r_last_hidden_states), dim=-1)
        # t_last_hidden_states = torch.cat(tuple(t_last_hidden_states), dim=-1)
        

        h_last_hidden_states = torch.mean(h_last_hidden_states, dim=1)
        r_last_hidden_states = torch.mean(r_last_hidden_states, dim=1)
        t_last_hidden_states = torch.mean(t_last_hidden_states, dim=1)


            # h_last_hidden_states = move_to_cpu(h_last_hidden_states)
            # r_last_hidden_states = move_to_cpu(r_last_hidden_states)
            # t_last_hidden_states = move_to_cpu(t_last_hidden_states)

            

        # embeddings = move_to_cuda(torch.stack((h_last_hidden_states, r_last_hidden_states, t_last_hidden_states), dim=1))
        embeddings = torch.stack((h_last_hidden_states, r_last_hidden_states, t_last_hidden_states))
        embeddings = torch.permute(embeddings, (1, 0, 2))

        # logger.info('batch size: {}'.format(embeddings.shape))
        # return move_to_cuda(embeddings)
        return (embeddings, h_ids, t_ids) 
    

    def get_ent_emb(self, examples, ent_ids: set) -> dict:
        logger.info('***Geting embedding***')
        features = convert_examples_to_features(examples, self.tokenizer, max_word_len=self.max_word_len)
        t_token_ids = []
        t_ids = []
        for f in features:
            if f['t_id'] in ent_ids:
                continue
            t_token_ids.append(f['t_token_id'])
            ent_ids.add(f['t_id'])
            t_ids.append(f['t_id'])

        t_token_ids = torch.tensor(t_token_ids, dtype=torch.long)
        if self.use_cuda:
            t_token_ids = move_to_cuda(t_token_ids)
        # print(h_token_ids.shape)

        # segment_tensor = torch.tensor([[1] * len(h_token_ids)] * len(examples))
        with torch.no_grad():
            t_last_hidden_states = self.t_model(t_token_ids)[2][-2]
            t_last_hidden_states = torch.mean(t_last_hidden_states, dim=1)

        # embeddings = move_to_cuda(t_last_hidden_states)

        # logger.info('batch size: {}'.format(embeddings.shape))
        return (t_last_hidden_states, t_ids)

    

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


        token_h = _truncate_and_padding_embedding(token_h, max_len=max_word_len)
        token_r = _truncate_and_padding_embedding(token_r, max_len=max_word_len)
        token_t = _truncate_and_padding_embedding(token_t, max_len=max_word_len)

        tokenized_h = tuple(tokenizer.convert_tokens_to_ids(token_h))
        tokenized_r = tuple(tokenizer.convert_tokens_to_ids(token_r))
        tokenized_t = tuple(tokenizer.convert_tokens_to_ids(token_t))


        features.append({
            'h_token_id': tokenized_h,
            'r_token_id': tokenized_r,
            't_token_id': tokenized_t,
            'h_id': example.h_id,
            't_id': example.t_id
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
            tail_desc=ent_descr[data[i]['tail_id']],
            head_id=data[i]['head_id'],
            tail_id=data[i]['tail_id']
        ))
        if inverse:
            examples.append(Example(
                # guid=f'{task}-{i}-inv',
                head=data[i]['tail'],
                relation='inverse ' + data[i]['relation'],
                tail=data[i]['head'],
                head_desc=ent_descr[data[i]['tail_id']],
                tail_desc=ent_descr[data[i]['head_id']],
                head_id=data[i]['tail_id'],
                tail_id=data[i]['head_id']
            ))
        data[i] = None
    return examples


# def collate(batch_data, task, pb=None) -> list:
#     '''
#     pos_samp: (batch_size, 3, 768)
#     neg_samp: (batch_size, 1, 768)
#     '''
#     embedder = get_bert_embedder()
#     embs = embedder.get_bert_embeddings(batch_data, task)

#     batch = []
#     for triple in embs:
#         neg_samp = None
#         # Self negatives
#         h = torch.unsqueeze(torch.unsqueeze(triple[0, :], 0), 0)
#         if neg_samp is None:
#             neg_samp = h
#         else:
#             neg_samp = torch.cat((neg_samp, h))
#         # Pre-batch
#         if pb is not None: # 
#             pb_tail = torch.cat([i['pos'] for i in pb])[:, 2, :]
#             pb_tail = torch.unsqueeze(pb_tail, dim=1)
#             neg_samp = torch.cat((neg_samp, pb_tail))
#         # In-batch
#         ts = torch.unsqueeze(embs[:, 2, :][rowwise_in(embs[:, 2, :], torch.unsqueeze(triple[2, :], 0))], dim=1)
#         batch.append({'pos': torch.unsqueeze(triple, 0), 'neg': neg_samp})
#     return batch

def collate(batch_data, task):
    embedder = get_bert_embedder()
    return embedder.get_bert_embeddings(batch_data, task)




def collate_entity(batch_data, ent_ids) -> list:
    embedder = get_bert_embedder()
    return embedder.get_ent_emb(batch_data, ent_ids)


class Collator:
    def __init__(self):
        self.prebatch = None
    def __call__(self, batch_data, task) -> Any:
        if self.prebatch is None:
            collated_data = collate(batch_data, task)
            self.prebatch = collated_data
            return collated_data
        else:
            collated_data = collate(batch_data, task, self.prebatch)
            self.prebatch = collated_data
            return collated_data


class TestCollator:
    def __init__(self, all_triples):
        self.all_triples, self.ent_ids = all_triples
        logger.info(self.all_triples.shape)

    def __call__(self, batch_data, task):
        embedder = get_bert_embedder()
        embs = embedder.get_bert_embeddings(batch_data, task)
        batch = []
        for triple in embs:
            batch.append({'pos': torch.unsqueeze(triple, 0), 'neg': self.all_triples})
        return batch
        

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
    

def get_all_ent_ids():
    global ALL_ENT_IDS
    if ALL_ENT_IDS is not None:
        return ALL_ENT_IDS
    paths=[
        main_args.train_path,
        main_args.test_path,
        main_args.valid_path
    ]
    ent_ids = []
    for path in paths:
        examples = load_data(path, inverse=False)
        for ex in examples:
            ent_ids.append(ex.t_id)
            ent_ids.append(ex.h_id)
    ALL_ENT_IDS = list(set(ent_ids))
    return ALL_ENT_IDS