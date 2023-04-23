import json
from logger_config import logger
import os
from transformers import BertModel, BertTokenizer
import torch
from utils import move_to_cuda, move_to_cpu
from functools import partial
from torch.utils.data import Dataset, DataLoader
from arguments import args



def load_data(path):
    '''Load data and create examples'''

    assert os.path.exists(path)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info('Load {} triplets from {}'.format(len(data), path))

    ent = set()
    for i in range(len(data)):
        ent.add(data[i]['head'])
        ent.add(data[i]['tail'])

    return list(ent)


def _truncate_and_padding_embedding(word: list, max_len) -> torch.tensor:
    '''Truncate or add padding to each entity/relation embedding'''
    if len(word) > max_len:
        return word[:max_len]
    elif len(word) < max_len:
        return word + ['[PAD]'] * (max_len - len(word))
    return word


def get_features(word, tokenizer: BertTokenizer): 
    token = tokenizer.tokenize(word)
    token = _truncate_and_padding_embedding(token, args.max_word_len)
    token_ids = tokenizer.convert_tokens_to_ids(token)
    return token_ids


def collate(batch, tokenizer: BertTokenizer):
    new_batch = []
    for item in batch:
        token = tokenizer.tokenize(item)
        token = _truncate_and_padding_embedding(token, args.max_word_len)
        token_ids = tokenizer.convert_tokens_to_ids(token)
        new_batch.append(token_ids)
    return new_batch 


class EntitiesDataset(Dataset):
    def __init__(self):
        self.entities = list(set(load_data(args.train_path) + load_data(args.test_path) + load_data(args.valid_path)))
        self.data_len = len(self.entities)
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        return self.entities[index]
    

def get_entities(model):
    logger.info('***Getting all entities embeddings***')
    entities = list(set(load_data(args.train_path) + load_data(args.test_path) + load_data(args.valid_path)))

    # if args.use_cuda:
    #     model = model.to('cuda')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    embeddings = []
    # batch_size = 64
    # batch = []
    token_ids = list(map(partial(get_features, tokenizer=tokenizer), entities))
    logger.info('***feature shape: {}'.format(len(token_ids)))
    logger.info('***token_id shape: {}'.format(len(token_ids[0])))

    ent_dataset = EntitiesDataset()

    data_loader = DataLoader(
        dataset=ent_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate, tokenizer=tokenizer)
    )

    for i , batch in enumerate(data_loader):
        logger.info('***Getting embeddings - batch {} ***'.format(i))
        with torch.no_grad():
            features = torch.tensor(batch, dtype=torch.long)
            if args.use_cuda:
                features = move_to_cuda(features)
                emb = model(features)[0]
                logger.info(emb.shape)
                emb = move_to_cpu(emb)
                logger.info('Embeddings shape {}'.format(emb.shape))
            else:
                emb = model(features)[0]
            emb = torch.mean(emb, dim=1)
            
            embeddings.append(emb)

    
    embeddings = torch.cat(embeddings, dim=0)
    torch.save(embeddings, 'entities.pt')
    return embeddings