from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
from torch.utils.data import Dataset
import torch
from loss import NTXentLoss
from pytorch_metric_learning import losses
from torch.nn import CrossEntropyLoss


class DSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=20,
                num_beams=20,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=20,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)

            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 20, -1)

        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


class distillDSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, num_beams, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.num_beams = num_beams
        self.first_loss_fct = CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False):
        # print(inputs)
        output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['supervised_labels'])
        loss = output.loss
        # logits = output.logits[:, :inputs['labels'].size(-1), :]
        # first_loss = self.first_loss_fct(logits.reshape(-1, logits.size(-1)), inputs['labels'].view(-1))
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            batch_beams = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=100,
                num_beams=self.num_beams,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                num_return_sequences=1,
                early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            # print(batch_beams.shape)
            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 1, -1)
            # print(batch_beams, batch_beams.shape)
        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor



class distillConsDSITrainer(Trainer):
        def __init__(self, restrict_decode_vocab, id_max_length, num_beams, **kwds):
            super().__init__(**kwds)
            self.restrict_decode_vocab = restrict_decode_vocab
            self.id_max_length = id_max_length
            self.num_beams = num_beams
            self.softmax_loss = losses.NTXentLoss(temperature=0.07)
            self.loss_fn = losses.CrossBatchMemory(
                loss=self.softmax_loss, embedding_size=768, memory_size=4096
            )

        def compute_loss(self, model, inputs, return_outputs=False):
            input_cls = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                              labels=inputs['supervised_labels']).encoder_last_hidden_state[:, 0]
            ori_cls = model(input_ids=inputs['ori_ids']['input_ids'],
                            attention_mask=inputs['ori_ids']['attention_mask'],
                            labels=inputs['supervised_labels']).encoder_last_hidden_state[:, 0]
            loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                         labels=inputs['supervised_labels']).loss
            # print(input_cls.shape, ori_cls.shape, len(inputs['int_ids']))
            # softmax_loss = self.softmax_loss(input_cls, ori_cls, inputs['int_ids'])
            embeddings = torch.cat([input_cls, ori_cls])
            labels = torch.tensor(inputs['int_ids'], device=input_cls.device)
            labels = torch.cat([labels, labels])
            # enqueue_idx = torch.arange(input_cls.size(0), input_cls.size(0) * 2)
            enqueue_idx = torch.arange(input_cls.size(0) * 2)
            softmax_loss = self.loss_fn(embeddings, labels)
            if return_outputs:
                return loss + 0.5 * softmax_loss, [None, None]  # fake outputs
            return loss + 0.5 * softmax_loss

        def prediction_step(
                self,
                model: nn.Module,
                inputs: Dict[str, Union[torch.Tensor, Any]],
                prediction_loss_only: bool,
                ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            model.eval()
            # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
            inputs['labels'] = inputs['labels'].to(self.args.device)
            with torch.no_grad():
                # Greedy search
                # doc_ids = model.generate(
                #     inputs['input_ids'].to(self.args.device),
                #     max_length=20,
                #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                #     early_stopping=True,)

                # Beam search
                batch_beams = model.generate(
                    inputs['input_ids'].to(self.args.device),
                    max_length=100,
                    num_beams=self.num_beams,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=1,
                    early_stopping=True, )

                if batch_beams.shape[-1] < self.id_max_length:
                    batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
                # print(batch_beams.shape)
                inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

                batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 1, -1)
                # print(batch_beams, batch_beams.shape)
            return (None, batch_beams, inputs['labels'])

        def _pad_tensors_to_max_len(self, tensor, max_length):
            if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
                # If PAD token is not defined at least EOS token has to be defined
                pad_token_id = (
                    self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                )
            else:
                if self.model.config.pad_token_id is not None:
                    pad_token_id = self.model.config.pad_token_id
                else:
                    raise ValueError(
                        "Pad_token_id must be set in the configuration of the model, in order to pad tensors")
            tensor[tensor == -100] = self.tokenizer.pad_token_id
            padded_tensor = pad_token_id * torch.ones(
                (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
            )
            padded_tensor[:, : tensor.shape[-1]] = tensor
            return padded_tensor


class embeddingdistillConsDSITrainer(Trainer):
    def __init__(self, restrict_decode_vocab, id_max_length, num_beams, predict_index, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab
        self.id_max_length = id_max_length
        self.num_beams = num_beams
        self.predict_index = predict_index
        self.softmax_loss = losses.NTXentLoss(temperature=0.07)
        self.loss_fn = losses.CrossBatchMemory(
                                    loss=self.softmax_loss, embedding_size=768, memory_size=1028
                                                )

    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        # input_cls = torch.max((output.encoder_last_hidden_state * inputs['attention_mask'].unsqueeze(-1)), axis=1).values
        loss = output.loss
        # print(input_cls.shape, ori_cls.shape, len(inputs['int_ids']))
        # softmax_loss = self.softmax_loss(input_cls, ori_cls, inputs['int_ids'])

        # cons_embs=torch.tensor(inputs['emb'], device=input_cls.device)
        # embeddings = torch.cat([input_cls, cons_embs])
        # labels = torch.tensor(inputs['qids'], device=input_cls.device)
        # labels = torch.cat([labels, labels])

        # enqueue_idx = torch.arange(input_cls.size(0), input_cls.size(0) * 2)
        # enqueue_idx = torch.arange(input_cls.size(0)*2)
        # softmax_loss = self.loss_fn(embeddings, labels)
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        inputs['labels'] = inputs['labels'].to(self.args.device)
        with torch.no_grad():
            # Greedy search
            # doc_ids = model.generate(
            #     inputs['input_ids'].to(self.args.device),
            #     max_length=20,
            #     prefix_allowed_tokens_fn=self.restrict_decode_vocab,
            #     early_stopping=True,)

            # Beam search
            if self.predict_index:
                batch_beams = model.generate(
                    inputs['input_ids'].to(self.args.device),
                    max_length=20,
                    num_beams=20,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=20,
                    early_stopping=True, )
            else:
                batch_beams = model.generate(
                    inputs['input_ids'].to(self.args.device),
                    max_length=100,
                    num_beams=self.num_beams,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=1,
                    early_stopping=True, )

            if batch_beams.shape[-1] < self.id_max_length:
                batch_beams = self._pad_tensors_to_max_len(batch_beams, self.id_max_length)
            # print(batch_beams.shape)
            inputs['labels'] = self._pad_tensors_to_max_len(inputs['labels'], self.id_max_length)

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], 1, -1)
            # print(batch_beams, batch_beams.shape)
        return (None, batch_beams, inputs['labels'])

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")
        tensor[tensor == -100] = self.tokenizer.pad_token_id
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
