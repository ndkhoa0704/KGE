from transformers import BertModel, BertTokenizerFast
from torch.utils.data import Dataset
import json
import os
from logger_config import logger
import torch


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

        # if i < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in token_seq]))
        #     logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

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


def _truncate_and_padding_embedding(batch: torch.tensor, max_len: int=7) -> torch.tensor:
    '''Truncate or add padding to each entity/relation embedding'''
    if batch.size()[1] > max_len:
        return batch[:,:max_len,:]
    elif batch.size()[1] < max_len:
        return torch.cat((batch, torch.zeros(batch.size()[0], max_len - batch.size()[1], batch.size(2))), dim=1)
    
    return batch


class BertEmbedder:
    def __init__(self,
                 pretrained_weights='bert-base-uncased',
                 tokenizer_class=BertTokenizerFast,
                 model_class=BertModel,
                 max_seq_len=20,
                 max_word_len=7):
        super().__init__()
        self.pretrained_weights = pretrained_weights
        self.tokenizer_class = tokenizer_class
        self.model_class = model_class
        self.tokenizer = self.tokenizer_class.from_pretrained(pretrained_weights)
        self.model = self.model_class.from_pretrained(pretrained_weights).cuda()
        self.max_seq_len = max_seq_len
        self.max_word_len = max_word_len
        # tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # model = BertModel.from_pretrained(pretrained_weights)

    def get_bert_embeddings(self, examples: list[Example]) -> dict:
        features = convert_examples_to_features(examples, self.tokenizer)

        all_token_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long).cuda()
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        logger.info('***Geting embedding***')
        last_hidden_states = self.model(all_token_ids)[0].cpu()  # Models outputs are now tuples

        embeddings = {
            'head': [],
            'relation': [],
            'tail': []
        }
        
        for i in range(len(examples)):
            head_emb = last_hidden_states[:,features[i].token_indices['head'][0]:features[i].token_indices['head'][1],:]
            head_emb = _truncate_and_padding_embedding(head_emb)
            embeddings['head'].append(head_emb)

            relation_emb = last_hidden_states[:,features[i].token_indices['relation'][0]:features[i].token_indices['relation'][1],:]
            relation_emb = _truncate_and_padding_embedding(relation_emb)
            embeddings['relation'].append(relation_emb)

            tail_emb = last_hidden_states[:,features[i].token_indices['tail'][0]:features[i].token_indices['tail'][1]:, ]
            tail_emb = _truncate_and_padding_embedding(tail_emb)
            embeddings['tail'].append(tail_emb)

        embeddings['head'] = torch.stack(embeddings['head']).cuda()
        embeddings['relation'] = torch.stack(embeddings['relation']).cuda()
        embeddings['tail'] = torch.stack(embeddings['tail']).cuda()

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