from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset
import json
import os
from logger_config import logger
import torch
from copy import deepcopy
from utils import move_to_cuda
from arguments import args



BERT_EMBEDDER = None

def get_bert_embedder(use_cuda, mode):
    global BERT_EMBEDDER
    if BERT_EMBEDDER is None:
        BERT_EMBEDDER = BertEmbedder(use_cuda=use_cuda, mode=mode)
    return BERT_EMBEDDER


class Example:
    '''
    Store textual triples
    '''
    def __init__(self, guid, head, relation, tail):
        self.guid = guid
        self.h = head
        self.r = relation
        self.t = tail

    @property
    def head(self):
        return _parse_entity(self.h)
    
    @property
    def relation(self):
        return self.r
    
    @property
    def tail(self):
        return _parse_entity(self.t)


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


def _truncate_and_padding_embedding(word: list, max_len) -> torch.tensor:
    '''Truncate or add padding to each entity/relation embedding'''
    if len(word) > max_len:
        return word[:max_len]
    elif len(word) < max_len:
        return word + ['[PAD]'] * (max_len - len(word))
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
        token_a = tokenizer.tokenize(example.head)
        token_b = tokenizer.tokenize(example.relation)
        token_c = tokenizer.tokenize(example.tail)

        token_a = _truncate_and_padding_embedding(token_a, max_len=max_word_len)
        token_b = _truncate_and_padding_embedding(token_b, max_len=max_word_len)
        token_c = _truncate_and_padding_embedding(token_c, max_len=max_word_len)

        triple = token_a + token_b + token_c

        tokens = tokenizer.convert_tokens_to_ids(triple)

        features.append({
            'h_token_id': tokens[:max_word_len],
            'r_token_id': tokens[max_word_len:2*max_word_len],
            't_token_id': tokens[2*max_word_len:],
            'hr_token_id': tokens[:2*max_word_len],
            'rt_token_id': tokens[max_word_len:],
        })

        examples[i] = None

    return features


def _parse_entity(entity: str) -> str:
    return entity.replace('_', ' ').strip()


class BertEmbedder:
    def __init__(
        self,
        pretrained_weights='bert-base-uncased',
        tokenizer_class=BertTokenizer,
        model_class=BertModel,
        max_seq_len=20,
        mode='forward',
        max_word_len=7,
        use_cuda=False
):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        print(self.tokenizer_class)
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.hr_model = self.model_class.from_pretrained(pretrained_weights).cuda()
        self.t_model = deepcopy(self.hr_model)
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        self.mode = mode
        self.use_cuda = use_cuda
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)


    def get_bert_embeddings(self, examples) -> dict:
        features = convert_examples_to_features(examples, self.tokenizer)
        
        if self.mode == 'forward':
            hr_token_ids = torch.tensor([f['hr_token_id'] for f in features], dtype=torch.long)
            t_token_ids = torch.tensor([f['t_token_id'] for f in features], dtype=torch.long)
        elif self.mode == 'backward':
            hr_token_ids = torch.tensor([f['rt_token_id'] for f in features], dtype=torch.long)
            t_token_ids = torch.tensor([f['t_token_id'] for f in features], dtype=torch.long)
        elif self.mode == 'all':
            hr_token_ids = torch.cat(
                (torch.tensor([f['hr_token_id'] for f in features], dtype=torch.long),
                torch.tensor([f['rt_token_id'] for f in features], dtype=torch.long)),
                dim=0
            )
            t_token_ids = torch.cat(
                (torch.tensor([f['t_token_id'] for f in features], dtype=torch.long),
                torch.tensor([f['t_token_id'] for f in features], dtype=torch.long)),
                dim=0
            )


        # all_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
        # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if self.use_cuda:
            hr_token_ids = move_to_cuda(hr_token_ids)
            t_token_ids = move_to_cuda(t_token_ids)

        logger.info('***Geting embedding***')
        # last_hidden_states = self.model(all_token_ids, all_attention_mask)[0].cpu()  # Models outputs are now tuples
        # if self.mode == 'forward': 
        #     hr_last_hidden_states = self.hr_model(hr_token_ids)[0]  # Models outputs are now tuples
        #     t_last_hidden_states = self.t_model(t_token_ids)[0]  # Models outputs are now tuples

        hr_last_hidden_states = self.hr_model(hr_token_ids)[0]  # Models outputs are now tuples
        t_last_hidden_states = self.t_model(t_token_ids)[0]
        h_emb = []
        r_emb = []
        t_emb = []

        for i in range(len(features)):
            # Models outputs are now tuples
            h_size = len(features[i]['h_token_id'])
            r_size = len(features[i]['r_token_id'])
            t_size = len(features[i]['t_token_id'])
            h_emb.append(torch.mean(hr_last_hidden_states[0, :h_size, :], dim=0))
            r_emb.append(torch.mean(hr_last_hidden_states[0, h_size:h_size+r_size, :], dim=0))
            t_emb.append(torch.mean(t_last_hidden_states[0, :t_size, :], dim=0))
        
        
        embeddings = {
            'head': h_emb,
            'relation': r_emb,
            'tail': t_emb
        }

        embeddings['head'] = torch.stack(embeddings['head']).cuda()
        embeddings['relation'] = torch.stack(embeddings['relation']).cuda()
        embeddings['tail'] = torch.stack(embeddings['tail']).cuda()

        # logger.info('batch size: {}'.format())
        return embeddings


def load_data(path, task, inverse=False):
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
                relation='inverse ' + data[i]['relation'],
                tail=data[i]['head']  
            ))
        data[i] = None

    return examples


def collate(batch_data, mode) -> list:
    embedder = get_bert_embedder(mode=mode, use_cuda=args.use_cuda)
    return embedder.get_bert_embeddings(batch_data)


class DataSet(Dataset):
    def __init__(self, path, task, *args, **kwargs):
        self.examples = load_data(path, task, *args, **kwargs)
        self.data_len = len(self.examples)
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.examples[index]
    

