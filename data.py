from dataclasses import dataclass
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import pickle
import numpy as np
import random
random.seed(10)

class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
    ):
        self.train_data = datasets.load_dataset(
            'json',
            data_files=path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            for id in str(data['text_id']).split(','):
                self.valid_ids.add(id)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        input_ids = self.tokenizer(data['text'],
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


class IndexingTrainPRFDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
            remove_first=False,
    ):
        self.path_to_data = path_to_data.split(',')
        self.train_data = datasets.load_dataset(
            'json',
            data_files=self.path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.remove_first = remove_first
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            for id in str(data['text_id']).split(','):
                self.valid_ids.add(id)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        if self.remove_first:
            text = data['text'] + ', prf_id: ' + data['prf_id'][6:].replace(',', ' ')
        else:
            text = data['text'] + ', prf_id: ' + data['prf_id'].replace(',', ' ')
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['text_id'])


class IndexingTrainDisltillDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
            is_training=False,
            gold_label=False,
            with_prf=False,
    ):
        self.path_to_data = path_to_data.split(',')
        self.train_data = datasets.load_dataset(
            'json',
            data_files=self.path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.is_training = is_training
        self.gold_label = gold_label
        self.with_prf = with_prf
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            for id in str(data['text_id']).split(','):
                self.valid_ids.add(id)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        if self.with_prf:
            text = data['text'] + ', prf_id: ' + data['prf_id'].replace(',', ' ')
        else:
            text = data['text']
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        # print(self.tokenizer(text,
        #                            return_tensors="pt",
        #                            truncation='only_first',
        #                            max_length=self.max_length).input_ids)
        if self.is_training:
            prf_ids = data['label_id'] if self.with_prf else data['prf_id']
            if self.gold_label:
                prf_id_list = prf_ids.split(',')
                if data['text_id'] in prf_id_list:
                    prf_id_list.remove(data['text_id'])
                return input_ids, str(data['text_id']), str(data['text_id']) + ' ' + ' '.join(prf_id_list)
            else:
                return input_ids, str(data['text_id']), str(data['text_id'])+' '+str(prf_ids.replace(',', ' '))
                # return input_ids, str(data['text_id']), str(prf_ids.replace(',', ' '))
        else:
            return input_ids, str(data['text_id']), str(data['text_id'])


class IndexingTrainDisltillDatasetCons(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            remove_prompt=False,
            is_training=False,
            gold_label=False,
            with_prf=False,
    ):
        self.path_to_data = path_to_data.split(',')
        self.train_data = datasets.load_dataset(
            'json',
            data_files=self.path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.is_training = is_training
        self.gold_label = gold_label
        self.with_prf = with_prf
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        for data in tqdm(self.train_data):
            for id in str(data['text_id']).split(','):
                self.valid_ids.add(id)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
        if self.with_prf:
            text = data['text'] + ', prf_id: ' + data['prf_id'].replace(',', ' ')
        else:
            text = data['text']
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]

        if self.is_training:
            ori_ids = self.tokenizer(data['ori_text'],
                                     return_tensors="pt",
                                     truncation='only_first',
                                     max_length=self.max_length).input_ids[0]
            prf_ids = data['label_id'] if self.with_prf else data['prf_id']
            if self.gold_label:
                prf_id_list = prf_ids.split(',')
                if data['text_id'] in prf_id_list:
                    prf_id_list.remove(data['text_id'])
                return input_ids, str(data['text_id']), str(data['text_id']) + ' ' + ' '.join(prf_id_list), ori_ids
            else:
                return input_ids, str(data['text_id']), str(data['text_id'])+' '+str(prf_ids.replace(',', ' ')), ori_ids
        else:
            return input_ids, str(data['text_id']), str(data['text_id']), input_ids

class EmbeddingIndexingTrainDisltillDatasetCons(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
            query_emb_dir='',
            doc_emb_dir='',
            remove_prompt=False,
            is_training=False,
            predict_index=False,
    ):
        self.path_to_data = path_to_data.split(',')
        self.train_data = datasets.load_dataset(
            'json',
            data_files=self.path_to_data,
            ignore_verifications=False,
            cache_dir=cache_dir
        )['train']

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.remove_prompt = remove_prompt
        self.is_training = is_training
        self.total_len = len(self.train_data)
        self.valid_ids = set()
        self.predict_index = predict_index
        for data in tqdm(self.train_data):
            for id in str(data['text_id']).split(','):
                self.valid_ids.add(id)
        # if self.is_training:
        #     query_f = open(query_emb_dir, 'rb')
        #     self.query_embedding_dict = pickle.load(query_f)
        #     doc_f = open(doc_emb_dir, 'rb')
        #     self.doc_embedding_dict = pickle.load(doc_f)
    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        if self.remove_prompt:
            data['text'] = data['text'][9:] if data['text'].startswith('Passage: ') else data['text']
            data['text'] = data['text'][10:] if data['text'].startswith('Question: ') else data['text']
            if self.predict_index:
                data['text'] = '<extra_id_0> ' + data['text']
            else:
                data['text'] = '<extra_id_1> ' + data['text']
        if self.is_training:
            prelist = ['<extra_id_0> ', '<extra_id_1> ']
            pre = random.choices(prelist, weights=[1, 1])
            data['text'] = pre[0] + data['text']
        text = data['text']
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]

        if self.is_training:
            emb = np.array(random.sample(range(1000),768)).astype(np.float32)
            # emb = self.query_embedding_dict[data['qid']] if data['text'].startswith('<extra_id_1>') \
            #       else self.doc_embedding_dict[data['qid']]
            prf_ids = data['prf_id']
            label = str(data['text_id'])+' '+str(prf_ids.replace(',', ' ')) if data['text'].startswith('<extra_id_1>') \
                else str(data['text_id'])
            qid = '100'+str(data['text_id']) if data['text'].startswith('<extra_id_1>') \
                else str(data['text_id'])
            return input_ids, qid, label, emb
        else:
            return input_ids, str(data['text_id']), str(data['text_id']), np.array(random.sample(range(1000),768)).astype(np.float32)

class GenerateDataset(Dataset):
    lang2mT5 = dict(
        ar='Arabic',
        bn='Bengali',
        fi='Finnish',
        ja='Japanese',
        ko='Korean',
        ru='Russian',
        te='Telugu'
    )

    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                if 'xorqa' in path_to_data:
                    docid, passage, title = data.split('\t')
                    for lang in self.lang2mT5.values():
                        self.data.append((docid, f'Generate {lang} question: {title}</s>{passage}'))
                elif 'msmarco' or 'NQ' in path_to_data:
                    docid, passage = data.split('\t')
                    self.data.append((docid, f'{passage}'))
                else:
                    raise NotImplementedError(f"dataset {path_to_data} for docTquery generation is not defined.")

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation='only_first',
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(docid)


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs

@dataclass
class DistillIndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        supervised = [x[2] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        supervised_labels = self.tokenizer(
            supervised, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        supervised_labels[supervised_labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['supervised_labels'] = supervised_labels
        return inputs

@dataclass
class DistillIndexingConsCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        ori_ids = [{'input_ids': x[3]} for x in features]
        docids = [x[1] for x in features]
        supervised = [x[2] for x in features]
        inputs = super().__call__(input_ids)
        ori_inputs = super().__call__(ori_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        supervised_labels = self.tokenizer(
            supervised, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        supervised_labels[supervised_labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['supervised_labels'] = supervised_labels
        inputs['int_ids'] = [int(docid.split(',')[0]) for docid in docids]
        inputs['ori_ids'] = ori_inputs
        return inputs

@dataclass
class EmbeddingDistillIndexingConsCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[2] for x in features]
        qids = [x[1] for x in features]
        embs = [x[3] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids


        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        inputs['qids'] = [int(qid.split(',')[0]) for qid in qids]
        inputs['emb'] = embs
        return inputs

@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels
