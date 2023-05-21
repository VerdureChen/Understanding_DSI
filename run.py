from data import IndexingTrainDataset, IndexingTrainPRFDataset, IndexingTrainDisltillDataset,\
    GenerateDataset, IndexingCollator, QueryEvalCollator, DistillIndexingCollator, DistillIndexingConsCollator,\
    IndexingTrainDisltillDatasetCons, EmbeddingDistillIndexingConsCollator, EmbeddingIndexingTrainDisltillDatasetCons
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    TrainerCallback,
    MT5Tokenizer,
    MT5TokenizerFast,
    MT5ForConditionalGeneration,
    HfArgumentParser,
    set_seed,
    logging
)
from trainer import DSITrainer, DocTqueryTrainer, distillDSITrainer, distillConsDSITrainer, embeddingdistillConsDSITrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
import datasets
from dataclasses import dataclass, field
from typing import Optional
import json
import os
from tqdm import tqdm
set_seed(313)

@dataclass
class RunArguments:
    model_name: str = field(default=None)
    model_path: Optional[str] = field(default=None)
    project_name: Optional[str] = field(default='DSI')
    max_length: Optional[int] = field(default=32)
    num_beams: Optional[int] = field(default=20)
    id_max_length: Optional[int] = field(default=20)
    remove_prompt: Optional[bool] = field(default=False)
    predict_index: Optional[bool] = field(default=False)
    remove_first: Optional[bool] = field(default=False)
    with_prf: Optional[bool] = field(default=False)
    train_file: str = field(default=None)
    valid_file: str = field(default=None)
    query_emb_dir: str = field(default=None)
    doc_emb_dir: str = field(default=None)
    task: str = field(default=None,  metadata={"help": "DSI, docTquery, generation"})
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)

def write_compute_metrics(tokenizer, valid_ids, valid_path, out_path):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        # k = 0
        with open(out_path, 'w') as f:
            ori_data = datasets.load_dataset('json', data_files=valid_path)['train']
            count=0
            for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
                flag_1 = 0
                flag_10 = 0
                rank_list = tokenizer.batch_decode(beams,
                                                   skip_special_tokens=True)
                label_ids = tokenizer.decode(label, skip_special_tokens=True)
                # filter out duplicates and invalid docids
                filtered_rank_list = []
                for docids in rank_list:
                    docids = docids.strip().split(' ')
                    # print(docids)
                    for docid in docids:
                        if docid not in filtered_rank_list and docid in valid_ids:
                            filtered_rank_list.append(docid)
                for label_id in label_ids.split(','):
                    hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
                    if len(hits) != 0:
                        if flag_10 == 0:
                            hit_at_10 += 1
                            flag_10 = 1
                        if flag_1 == 0:
                            if hits[0] == 0:
                                hit_at_1 += 1
                                flag_1 = 1
                text_id = ori_data[count]['text_id']
                text = ori_data[count]['text']
                if 'qid' in ori_data[count]:
                    qid = ori_data[count]['qid']
                    jitem = json.dumps({'qid': qid,
                                        'text_id': text_id,
                                        'text': text,
                                        'prf_id': ",".join(map(str, filtered_rank_list[:10], ))})
                else:
                    jitem = json.dumps({'text_id': text_id,
                                        'text': text,
                                        'prf_id': ",".join(map(str, filtered_rank_list[:10], ))})
                f.write(jitem + '\n')
                count += 1
            return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


def make_compute_metrics(tokenizer, valid_ids):
    def compute_metrics(eval_preds):
        hit_at_1 = 0
        hit_at_10 = 0
        # k = 0
        for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
            flag_1 = 0
            flag_10 = 0
            rank_list = tokenizer.batch_decode(beams,
                                               skip_special_tokens=True)
            label_ids = tokenizer.decode(label, skip_special_tokens=True)
            # filter out duplicates and invalid docids
            filtered_rank_list = []
            for docids in rank_list:
                docids = docids.strip().split(' ')
                # print(docids)
                for docid in docids:
                    if docid not in filtered_rank_list and docid in valid_ids:
                        filtered_rank_list.append(docid)
            for label_id in label_ids.split(','):
                hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
                if len(hits) != 0:
                    if flag_10 == 0:
                        hit_at_10 += 1
                        flag_10 = 1
                    if flag_1 == 0:
                        if hits[0] == 0:
                            hit_at_1 += 1
                            flag_1 = 1
        return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
    return compute_metrics


# def make_distill_compute_metrics(tokenizer, valid_ids):
#
#     def compute_metrics(eval_preds):
#         hit_at_1 = 0
#         hit_at_10 = 0
#         # k = 0
#         for beams, label in zip(eval_preds.predictions, eval_preds.label_ids):
#             flag_1 = 0
#             flag_10 = 0
#             rank_list = tokenizer.batch_decode(beams,
#                                                skip_special_tokens=True)
#             label_ids = tokenizer.decode(label, skip_special_tokens=True)
#             # filter out duplicates and invalid docids
#             filtered_rank_list = []
#             # print(rank_list, len(rank_list))
#             for docids in rank_list:
#                 docids = docids.strip().split(' ')
#                 # print(docids)
#                 for docid in docids:
#                     if docid not in filtered_rank_list and docid in valid_ids:
#                         filtered_rank_list.append(docid)
#             # if k < 10:
#             #     print(filtered_rank_list, label_id)
#             # k = k + 1
#             for label_id in label_ids.split(','):
#                 hits = np.where(np.array(filtered_rank_list)[:10] == label_id)[0]
#                 if len(hits) != 0:
#                     if flag_10 == 0:
#                         hit_at_10 += 1
#                         flag_10 = 1
#                     if flag_1 == 0:
#                         if hits[0] == 0:
#                             hit_at_1 += 1
#                             flag_1 = 1
#         return {"Hits@1": hit_at_1 / len(eval_preds.predictions), "Hits@10": hit_at_10 / len(eval_preds.predictions)}
#     return compute_metrics


def main():

    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()
    print(f'train_args:{training_args},\n'
          f'run_args:{run_args}')
    logging.set_verbosity_info()
    # We use wandb logger: https://wandb.ai/site.
    # if training_args.local_rank == 0:  # only on main process
    #     # Initialize wandb run
    #     wandb.login()
    #     wandb.init(project=run_args.project_name, name=training_args.run_name, dir='output/logs/wandb')

    if 'mt5' in run_args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = MT5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = MT5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')
    else:
        tokenizer = T5Tokenizer.from_pretrained(run_args.model_name, cache_dir='cache')
        fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_name, cache_dir='cache')
        if run_args.model_path:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')
        else:
            model = T5ForConditionalGeneration.from_pretrained(run_args.model_name, cache_dir='cache')

    if run_args.task == "docTquery":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        trainer = DocTqueryTrainer(
            do_generation=False,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
        )
        trainer.train()

    elif run_args.task == "DSI":
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)

        valid_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             remove_prompt=run_args.remove_prompt,
                                             tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        trainer.train(resume_from_checkpoint=True)
        # trainer.train()

    elif run_args.task == 'generation':
        generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

        trainer = DocTqueryTrainer(
            do_generation=True,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=QueryEvalCollator(
                tokenizer,
                padding='longest',
            ),
        )
        predict_results = trainer.predict(generate_dataset,
                                          top_k=run_args.top_k,
                                          num_return_sequences=run_args.num_return_sequences,
                                          max_length=run_args.q_max_length)
        with open(f"{run_args.valid_file}.q{run_args.num_return_sequences}.docTquery", 'w') as f:
            for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids),
                                                desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    query = fast_tokenizer.decode(tokens, skip_special_tokens=True)
                    jitem = json.dumps({'text_id': docid.item(), 'text': query})
                    f.write(jitem + '\n')

    elif run_args.task == 'gen_prf':
        train_dataset = IndexingTrainDataset(path_to_data=run_args.train_file,
                                             max_length=run_args.max_length,
                                             cache_dir='cache',
                                             tokenizer=tokenizer)
        fst_run_dataset = IndexingTrainDataset(path_to_data=run_args.valid_file,
                                               max_length=run_args.max_length,
                                               cache_dir='cache',
                                               remove_prompt=run_args.remove_prompt,
                                               tokenizer=tokenizer)
        # train_dataset = IndexingTrainPRFDataset(path_to_data=run_args.train_file,
        #                                         max_length=run_args.max_length,
        #                                         cache_dir='cache',
        #                                         tokenizer=tokenizer)
        #
        # fst_run_dataset = IndexingTrainPRFDataset(path_to_data=run_args.valid_file,
        #                                           max_length=run_args.max_length,
        #                                           cache_dir='cache',
        #                                           remove_prompt=run_args.remove_prompt,
        #                                           remove_first=run_args.remove_first,
        #                                           tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS

        ################################################################
        fl_path, fl_name = os.path.split(run_args.valid_file)
        # ori_data.to_json(f"temp/results/{fl_name}.withQgPRF")
        out_path = f'{fl_path}/results/{fl_name+"_"+run_args.model_name.split("/")[-2]}.withQgPRF'
        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=write_compute_metrics(fast_tokenizer, train_dataset.valid_ids, run_args.valid_file, out_path),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
        )
        predict_results = trainer.predict(fst_run_dataset,)
        print(f'metrics:{predict_results.metrics}')
        fl_path, fl_name = os.path.split(run_args.valid_file)
        with open(f"{fl_path}/results/QG.record", 'a') as f:
        # with open(f"temp/results/QG.record", 'a') as f:
            f.write(f'file:{run_args.model_name.split("/")[-2]+"_"+fl_name}, metrics:{predict_results.metrics}\n')
        # with open(f"{fl_path}/results/{fl_name}.withQGPRF", 'w') as f:
        #     ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)
        #     for beams, label, ori_line in tqdm(zip(predict_results.predictions, predict_results.label_ids, ori_data['train']),
        #                                        desc="Writing file"):
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = ori_line['text_id']
        #         ori_query = ori_line['text']
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docid in rank_list:
        #             if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                 filtered_rank_list.append(docid)
        #         # if len(filtered_rank_list)!=20:
        #         #     print(f'text_id:{label_id}, length:{len(filtered_rank_list)}, ranklist:{rank_list}')
        #         jitem = json.dumps({'text_id': label_id, 'text': ori_query,
        #                             'prf_id': ",".join(map(str, filtered_rank_list[:10],))})
        #         f.write(jitem + '\n')
            # for item in filtered_rank_lists:
            #     jitem = json.dumps(item)
            #     f.write(jitem + '\n')
        # ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)['train']
        #
        # def get_prf_dataset(examples, idxs):
        #     examples['prf_id'] = []
        #     for example, idx in zip(examples['text_id'], idxs):
        #         # print(example, idx, type(examples['text_id']))
        #         beams = predict_results.predictions[idx]
        #         label = predict_results.label_ids[idx]
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = example
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docid in rank_list:
        #             if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                 filtered_rank_list.append(docid)
        #         examples['prf_id'].append(",".join(map(str, filtered_rank_list[:10], )))
        #     return examples
        #
        # ori_data = ori_data.map(get_prf_dataset, batched=True, with_indices=True, num_proc=3, batch_size=10)
        # fl_path, fl_name = os.path.split(run_args.valid_file)
        # # ori_data.to_json(f"temp/results/{fl_name}.withQgPRF")
        # ori_data.to_json(f"{fl_path}/results/{fl_name}.withQgPRF")

    elif run_args.task == "DSI_prf":
        train_dataset = IndexingTrainPRFDataset(path_to_data=run_args.train_file,
                                                max_length=run_args.max_length,
                                                cache_dir='cache',
                                                tokenizer=tokenizer)

        valid_dataset = IndexingTrainPRFDataset(path_to_data=run_args.valid_file,
                                                max_length=run_args.max_length,
                                                cache_dir='cache',
                                                remove_prompt=run_args.remove_prompt,
                                                remove_first=run_args.remove_first,
                                                tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = DSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length
        )
        trainer.train()

    elif run_args.task == "DistillDSI":
        train_dataset = IndexingTrainDisltillDataset(path_to_data=run_args.train_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     tokenizer=tokenizer,
                                                     is_training=True,
                                                     with_prf=run_args.with_prf)

        valid_dataset = IndexingTrainDisltillDataset(path_to_data=run_args.valid_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     remove_prompt=run_args.remove_prompt,
                                                     tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = distillDSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DistillIndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids.union(valid_dataset.valid_ids)),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            num_beams=run_args.num_beams
        )
        trainer.train()

    elif run_args.task == "DistillDSI_cons":
        train_dataset = IndexingTrainDisltillDatasetCons(path_to_data=run_args.train_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     tokenizer=tokenizer,
                                                     is_training=True,
                                                     with_prf=run_args.with_prf)

        valid_dataset = IndexingTrainDisltillDatasetCons(path_to_data=run_args.valid_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     remove_prompt=run_args.remove_prompt,
                                                     tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = distillConsDSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DistillIndexingConsCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids.union(valid_dataset.valid_ids)),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            num_beams=run_args.num_beams
        )
        trainer.train()

    elif run_args.task == "EmbDistillDSI_cons":
        train_dataset = EmbeddingIndexingTrainDisltillDatasetCons(path_to_data=run_args.train_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     tokenizer=tokenizer,
                                                     is_training=True,
                                                     query_emb_dir=run_args.query_emb_dir,
                                                     doc_emb_dir=run_args.doc_emb_dir,
                                                                  )

        valid_dataset = EmbeddingIndexingTrainDisltillDatasetCons(path_to_data=run_args.valid_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     remove_prompt=run_args.remove_prompt,
                                                     tokenizer=tokenizer,
                                                     predict_index=run_args.predict_index)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################

        trainer = embeddingdistillConsDSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=EmbeddingDistillIndexingConsCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=make_compute_metrics(fast_tokenizer, train_dataset.valid_ids.union(valid_dataset.valid_ids)),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            num_beams=run_args.num_beams,
            predict_index=run_args.predict_index
        )
        trainer.train(resume_from_checkpoint=True)
        # trainer.train()

    elif run_args.task == 'gen_distill_prf':
        train_dataset = IndexingTrainDisltillDataset(path_to_data=run_args.train_file,
                                                     max_length=run_args.max_length,
                                                     cache_dir='cache',
                                                     tokenizer=tokenizer)

        fst_run_dataset = IndexingTrainDisltillDataset(path_to_data=run_args.valid_file,
                                                       max_length=run_args.max_length,
                                                       cache_dir='cache',
                                                       remove_prompt=run_args.remove_prompt,
                                                       tokenizer=tokenizer)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS

        ################################################################
        fl_path, fl_name = os.path.split(run_args.valid_file)
        out_path=f"{fl_path}/result/{fl_name}.{os.path.basename(os.path.dirname(run_args.model_name))}.withDistillPRF"
        trainer = distillDSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=DistillIndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=write_compute_metrics(fast_tokenizer, train_dataset.valid_ids, run_args.valid_file, out_path),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            num_beams=run_args.num_beams
        )
        predict_results = trainer.predict(fst_run_dataset,)
        print(f'metrics:{predict_results.metrics}')
        with open(f"{fl_path}/results/Distill.record", 'a') as f:
            f.write(f'file:{run_args.model_name.split("/")[-2]+"_"+fl_name}, metrics:{predict_results.metrics}\n')
        # with open(f"{run_args.valid_file}.withDistillPRF", 'w') as f:
        #     ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)
        #     for beams, label, ori_line in tqdm(zip(predict_results.predictions, predict_results.label_ids, ori_data['train']),
        #                                        desc="Writing file"):
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = ori_line['text_id']
        #         ori_query = ori_line['text']
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docids in rank_list:
        #             docids = docids.strip().split(' ')
        #             for docid in docids:
        #                 if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                     filtered_rank_list.append(docid)
        #         # if len(filtered_rank_list)!=20:
        #         #     print(f'text_id:{label_id}, length:{len(filtered_rank_list)}, ranklist:{rank_list}')
        #         # jitem = json.dumps({'text_id': label_id, 'text': ori_query,
        #         #                     'prf_id': ",".join(map(str, filtered_rank_list[:10],))})
        #         f.write('{text_id:' +label_id+ 'text:'+ ori_query+'prf_id:' + ",".join(map(str, filtered_rank_list[:10],))+'}\n')
        # ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)['train']
        # def get_prf_dataset(examples, idxs):
        #     # examples['prf_id'] = examples['prf_id'] if 'prf_id' in examples else []
        #     examples['prf_id'] = []
        #     for example, idx in zip(examples['text_id'], idxs):
        #         # print(example, idx, type(examples['text_id']))
        #         beams = predict_results.predictions[idx]
        #         label = predict_results.label_ids[idx]
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = example
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docids in rank_list:
        #             docids = docids.strip().split(' ')
        #             for docid in docids:
        #                 if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                     filtered_rank_list.append(docid)
        #         examples['prf_id'].append(",".join(map(str, filtered_rank_list[:10],)))
        #     return examples
        # ori_data = ori_data.map(get_prf_dataset, batched=True, with_indices=True, num_proc=5)
        # fl_path, fl_name = os.path.split(run_args.valid_file)
        # ori_data.to_json(f"{fl_path}/result/{fl_name}.{os.path.basename(os.path.dirname(run_args.model_name))}.withDistillPRF")

    elif run_args.task == 'gen_emb_prf':
        train_dataset = EmbeddingIndexingTrainDisltillDatasetCons(path_to_data=run_args.train_file,
                                                                  max_length=run_args.max_length,
                                                                  cache_dir='cache',
                                                                  tokenizer=tokenizer)

        fst_run_dataset = EmbeddingIndexingTrainDisltillDatasetCons(path_to_data=run_args.valid_file,
                                                                    max_length=run_args.max_length,
                                                                    cache_dir='cache',
                                                                    remove_prompt=run_args.remove_prompt,
                                                                    tokenizer=tokenizer,
                                                                    predict_index=run_args.predict_index)
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS

        ################################################################
        fl_path, fl_name = os.path.split(run_args.valid_file)
        out_path = f"{fl_path}/results/{fl_name}.{os.path.basename(os.path.dirname(run_args.model_name))}." \
                   f"{'index' if run_args.predict_index else 'retvl'}.withconsDistillPRF"
        trainer = embeddingdistillConsDSITrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=EmbeddingDistillIndexingConsCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=write_compute_metrics(fast_tokenizer, train_dataset.valid_ids, run_args.valid_file, out_path),
            restrict_decode_vocab=restrict_decode_vocab,
            id_max_length=run_args.id_max_length,
            num_beams=run_args.num_beams,
            predict_index=run_args.predict_index
        )
        predict_results = trainer.predict(fst_run_dataset,)
        print(f'metrics:{predict_results.metrics}')
        fl_path, fl_name = os.path.split(run_args.valid_file)
        with open(f"{fl_path}/results/MULTI.record", 'a') as f:
            f.write(f'file:{fl_name}, index:{str(run_args.predict_index)}, metrics:{predict_results.metrics}\n')
        # with open(f"{run_args.valid_file}.withDistillPRF", 'w') as f:
        #     ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)
        #     for beams, label, ori_line in tqdm(zip(predict_results.predictions, predict_results.label_ids, ori_data['train']),
        #                                        desc="Writing file"):
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = ori_line['text_id']
        #         ori_query = ori_line['text']
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docids in rank_list:
        #             docids = docids.strip().split(' ')
        #             for docid in docids:
        #                 if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                     filtered_rank_list.append(docid)
        #         # if len(filtered_rank_list)!=20:
        #         #     print(f'text_id:{label_id}, length:{len(filtered_rank_list)}, ranklist:{rank_list}')
        #         # jitem = json.dumps({'text_id': label_id, 'text': ori_query,
        #         #                     'prf_id': ",".join(map(str, filtered_rank_list[:10],))})
        #         f.write('{text_id:' +label_id+ 'text:'+ ori_query+'prf_id:' + ",".join(map(str, filtered_rank_list[:10],))+'}\n')
        # ori_data = datasets.load_dataset('json', data_files=run_args.valid_file)['train']
        # def get_prf_dataset(examples, idxs):
        #     # examples['prf_id'] = examples['prf_id'] if 'prf_id' in examples else []
        #     examples['prf_id'] = []
        #     for example, idx in zip(examples['text_id'], idxs):
        #         # print(example, idx, type(examples['text_id']))
        #         beams = predict_results.predictions[idx]
        #         label = predict_results.label_ids[idx]
        #         rank_list = tokenizer.batch_decode(beams,
        #                                            skip_special_tokens=True)
        #         label_id = tokenizer.decode(label, skip_special_tokens=True)
        #         ori_id = example
        #         assert str(label_id) == str(ori_id), "cannot match the original data and predict data."
        #         # filter out duplicates and invalid docids
        #         filtered_rank_list = []
        #         for docids in rank_list:
        #             docids = docids.strip().split(' ')
        #             for docid in docids:
        #                 if docid not in filtered_rank_list and docid in train_dataset.valid_ids:
        #                     filtered_rank_list.append(docid)
        #         examples['prf_id'].append(",".join(map(str, filtered_rank_list[:10],)))
        #     return examples
        # ori_data = ori_data.map(get_prf_dataset, batched=True, with_indices=True, num_proc=3, batch_size=10)
        # fl_path, fl_name = os.path.split(run_args.valid_file)
        # ori_data.to_json(f"{fl_path}/result/{fl_name}.{os.path.basename(os.path.dirname(run_args.model_name))}."
        #                  f"{'index' if run_args.predict_index else 'retvl'}.withconsDistillPRF")


    else:
        raise NotImplementedError("--task should be in 'DSI' or 'docTquery' or 'generation'")


if __name__ == "__main__":
    main()

