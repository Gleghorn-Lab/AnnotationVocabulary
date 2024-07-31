import argparse
import torch
import random
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from typing import *
from torchinfo import summary
from transformers import AutoTokenizer, EvalPrediction
from datasets import load_dataset
from functools import partial
from udev_public.data.data_utils import get_max_from_list_of_lists
from udev_public.models.camp.modeling_campv3 import CAMPv3 # GSM was originally called CAMPv3
from udev_public.models.camp.config_campv3 import CAMPv3Config # we are not releasing GSM right now
from udev_public.data.data_utils import process_column
from udev_public.trainers.huggingface import hf_trainer_from_args
from sklearn.metrics import accuracy_score


def comput_acc(preds, labels):
    preds, labels = preds.flatten(), labels.flatten()
    valid_indices = labels != -100
    preds = preds[valid_indices]
    labels = labels[valid_indices]
    return accuracy_score(labels, preds)


def compute_metrics_campv2_full(p: EvalPrediction):
    outputs = p.predictions
    at_logits, at_labels, tg_logits, tg_labels, tg_c_loss, ct_logits, ct_labels, l1_loss = outputs
    ct_acc = comput_acc(ct_logits, ct_labels)
    tg_acc = comput_acc(tg_logits, tg_labels)
    at_acc = comput_acc(at_logits, at_labels)
    return {
        'context_accuracy': ct_acc,
        'target_accuracy': tg_acc,
        'annotation_accuracy': at_acc,
        'l1_loss': l1_loss.mean(),
        'tg_c_loss': tg_c_loss.mean(),
    }


def compute_metrics_campv2_rep(p: EvalPrediction):
    outputs = p.predictions
    at_logits, at_labels, tg_logits, tg_labels, tg_c_loss = outputs
    tg_acc = comput_acc(tg_logits, tg_labels)
    at_acc = comput_acc(at_logits, at_labels)
    return {
        'target_accuracy': tg_acc,
        'annotation_accuracy': at_acc,
        'tg_c_loss': tg_c_loss.mean(),
    }


def compute_metrics_campv2_gen(p: EvalPrediction):
    outputs = p.predictions
    at_logits, at_labels, ct_logits, ct_labels = outputs
    ct_acc = comput_acc(ct_logits, ct_labels)
    at_acc = comput_acc(at_logits, at_labels)
    return {
        'context_accuracy': ct_acc,
        'annotation_accuracy': at_acc,
    }


class CampDatasetTest(TorchDataset):
    def __init__(self, data, seq_col, ann_col, ann_max_length=256, seq_max_length=512):
        self.anns = data[ann_col]
        self.seqs = data[seq_col]
        self.ann_max_length = ann_max_length
        self.seq_max_length = seq_max_length

    def avg(self):
        return sum(len(seq) for seq in self.seqs) / len(self.seqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:self.seq_max_length]
        ann = self.anns[idx]
        if len(ann) > self.ann_max_length:
            random.shuffle(ann)
            ann = ann[:self.ann_max_length]
            ann = sorted(ann)
        return seq, ann
    

class CampDatasetTrain(TorchDataset):
    def __init__(self, dataset, seq_col, ann_col, ann_max_length=256, seq_max_length=512, finetune=False):
        red_data = dataset['long_exp'] if finetune else dataset['train'] 
        exp_data = dataset['long_exp'] if finetune else dataset['exp']
        self.red_seqs, self.red_anns = red_data[seq_col], red_data[ann_col]
        self.exp_seqs, self.exp_anns = exp_data[seq_col], exp_data[ann_col]
        self.ann_max_length = ann_max_length
        self.seq_max_length = seq_max_length
        self.epoch = 0
        self.reset_epoch()

    def shuffle_data(self):
        red_data = list(zip(self.red_seqs, self.red_anns))
        exp_data = list(zip(self.exp_seqs, self.exp_anns))
        random.shuffle(red_data)
        random.shuffle(exp_data)
        self.red_seqs, self.red_anns = zip(*red_data)
        self.exp_seqs, self.exp_anns = zip(*exp_data)

    def reset_epoch(self):
        self.shuffle_data()
        self.seqs = list(self.red_seqs) + list(self.exp_seqs)
        self.anns = list(self.red_anns) + list(self.exp_anns)
        self.epoch += 1

    def avg(self):
        return sum(len(seq) for seq in self.seqs) / len(self.seqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][:self.seq_max_length]
        ann = self.anns[idx]
        if len(ann) > self.ann_max_length:
            ann = random.sample(ann, self.ann_max_length)
            ann = sorted(ann)
        
        if idx == len(self) - 1:
            self.reset_epoch()
        
        return seq, ann


class CampCollator:
    def __init__(
            self,
            tokenizer,
            base_tokenizer, # ankh tokenizer
            num_annotation_tokens=32000,
            annotation_mask_probability=0.15,
            context_mask_probability=0.30,
            sequence_mask_std=0.1,
    ):
        self.annotion_mask_probability = annotation_mask_probability
        self.num_annotation_tokens = num_annotation_tokens
        self.ann_vocab_size = num_annotation_tokens
        self.ann_pad_token_id = 0
        self.ann_cls_id = num_annotation_tokens + 10 - 3 # we added 10 in initalization
        self.ann_eos_id = num_annotation_tokens + 10  - 2
        self.ann_mask_token_id = self.ann_vocab_size + 10  - 1
        self.ann_special_token_ids = [self.ann_pad_token_id, self.ann_cls_id, self.ann_eos_id, self.ann_mask_token_id]

        self.base_tokenizer = base_tokenizer
        self.tokenizer = tokenizer
        self.context_mask_probability = context_mask_probability
        self.sequence_mask_std = sequence_mask_std
        self.seq_vocab_size = tokenizer.vocab_size
        self.seq_pad_token_id = tokenizer.pad_token_id
        self.seq_cls_id = tokenizer.cls_token_id
        self.seq_eos_id = tokenizer.eos_token_id
        self.seq_mask_token_id = tokenizer.mask_token_id
        self.seq_special_token_ids = [self.seq_pad_token_id, self.seq_cls_id, self.seq_eos_id, self.seq_mask_token_id]

    def __call__(self, examples):
        seqs = [example[0] for example in examples]
        anns = [example[1] for example in examples]

        ann_input_ids = [[self.ann_cls_id] + ann + [self.ann_eos_id] for ann in anns]
        ann_input_ids = self._pad_sequences(ann_input_ids)
        ann_labels = ann_input_ids.clone()
        ann_attention_mask = self._create_attention_mask(ann_input_ids)
        if self.annotion_mask_probability > 0:
            ann_input_ids, ann_labels = self._mask_tokens(
                inputs=ann_input_ids,
                labels=ann_labels,
                vocab_size=self.ann_vocab_size,
                mask_prob=self.annotion_mask_probability,
                mask_token_id=self.ann_mask_token_id,
                special_token_ids=self.ann_special_token_ids,
                seq=False
            )
        else:
            ann_labels[ann_input_ids.eq(self.ann_pad_token_id)] = -100

        base_tokens = self.base_tokenizer(seqs, return_tensors='pt', padding='longest', truncation=False)
        seq_tokens = self.tokenizer(seqs, padding='longest', truncation=False, return_tensors='pt')
        original_ids = seq_tokens.input_ids
        seq_attention_mask = seq_tokens.attention_mask
        context_input_ids, context_labels = self._mask_tokens(
            inputs=original_ids.clone(),
            labels=original_ids.clone(),
            vocab_size=self.seq_vocab_size,
            mask_prob=self.context_mask_probability,
            mask_token_id=self.seq_mask_token_id,
            special_token_ids=self.seq_special_token_ids,
            seq=True
        )
        target_input_ids = original_ids.clone()
        target_labels = original_ids.clone()
        target_labels[original_ids.eq(self.seq_pad_token_id)] = -100
        batch = {
            'base_input_ids': base_tokens.input_ids,
            'base_attention_mask': base_tokens.attention_mask,
            'context_input_ids': context_input_ids,
            'context_attention_mask': seq_attention_mask,
            'target_input_ids': target_input_ids,
            'target_attention_mask': seq_attention_mask,
            'annotation_input_ids': ann_input_ids,
            'annotation_attention_mask': ann_attention_mask,
            'context_labels': context_labels,
            'target_labels': target_labels,
            'annotation_labels': ann_labels,
            'labels': torch.ones(len(examples)) # so compute metrics works
        }
        return batch

    def _pad_sequences(self, sequences: List[List[int]]) -> torch.Tensor:
        max_length = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [self.ann_pad_token_id] * (max_length - len(seq)) for seq in sequences]
        return torch.tensor(padded_sequences, dtype=torch.long)

    def _create_attention_mask(self, input_ids: List[List[int]]) -> torch.Tensor:
        attention_mask =  (input_ids != self.ann_pad_token_id).long()
        return attention_mask

    def gen_prob(self, prob, min=0.05, max=1.0):
        prob = np.random.normal(prob, self.sequence_mask_std)
        return np.clip(prob, min, max)

    def _mask_tokens(self,
                     inputs,
                     labels,
                     vocab_size,
                     mask_prob,
                     mask_token_id,
                     special_token_ids,
                     seq=False):
        if seq:
            mask_prob = self.gen_prob(mask_prob)
        probability_matrix = torch.full(inputs.shape, mask_prob)
        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for token_id in special_token_ids:
            special_tokens_mask |= (inputs == token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = mask_token_id
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='lhallee/CAMP_test')
    parser.add_argument('--model_path', type=str, default='lhallee/campv3_gen_150_low')
    parser.add_argument('--context_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--target_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--annotation_path', type=str, default='lhallee/AT_new')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--context_mask', type=float, default=0.50)
    parser.add_argument('--ann_mask', type=float, default=0.15)
    parser.add_argument('--sequence_std', type=float, default=0.1)
    parser.add_argument('--at_ce_hyper', type=float, default=1.0)
    parser.add_argument('--tg_ce_hyper', type=float, default=1.0)
    parser.add_argument('--tg_cont_hyper', type=float, default=1.0)
    parser.add_argument('--ct_ce_hyper', type=float, default=1.0)
    parser.add_argument('--l1_hyper', type=float, default=1.0)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--seq_max_length', type=int, default=512)
    parser.add_argument('--ann_max_length', type=int, default=256)
    parser.add_argument('--contrastive_loss', type=str, default='mnr')
    parser.add_argument('--data_path', type=str, default='lhallee/new_annotation_vocab')
    parser.add_argument('--columns', nargs='+')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--finetune', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    if args.finetune:
        config = CAMPv3Config.from_pretrained(args.model_path, token=args.token)
        config.token = args.token
        model = CAMPv3.from_pretrained(args.model_path, config=config, token=args.token)
    else:
        config = CAMPv3Config(
            context_path=args.context_path,
            target_path=args.target_path,
            annotation_path=args.annotation_path,
            context=args.context,
            target=args.target,
            contrastive_loss=args.contrastive_loss,
            at_ce_hyper=args.at_ce_hyper,
            tg_ce_hyper=args.tg_ce_hyper,
            tg_cont_hyper=args.tg_cont_hyper,
            ct_ce_hyper=args.ct_ce_hyper,
            l1_hyper=args.l1_hyper,
            space_lambda1=args.lambda1,
            space_lambda2=args.lambda2,
            token=args.token)
        model = CAMPv3(config)
    model.config.token = None # remove HF token
    summary(model)

    data = load_dataset(args.data_path, token=args.token)

    if args.columns == None:
        seq_col, nlp_col = 'seqs', 'combined'
    else:
        seq_col, nlp_col = args.columns

    try:
        process = partial(process_column, col_name=nlp_col)    
        data = data.map(process)
    except:
        pass
    num_annotation_tokens = max(get_max_from_list_of_lists(data['train'][nlp_col]),
                                get_max_from_list_of_lists(data['exp'][nlp_col]),
                                get_max_from_list_of_lists(data['test'][nlp_col]))
    print(num_annotation_tokens)
    valid = data['test']
    train_dataset = CampDatasetTrain(
        data,
        seq_col,
        nlp_col,
        ann_max_length=args.ann_max_length,
        seq_max_length=args.seq_max_length,
        finetune=args.finetune
    )
    valid_dataset = CampDatasetTest(
        valid,
        seq_col,
        nlp_col,
        ann_max_length=args.ann_max_length,
        seq_max_length=args.seq_max_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.context_path, token=args.token)
    base_tokenizer = AutoTokenizer.from_pretrained('ElnaggarLab/ankh-base', token=args.token)
    data_collator = CampCollator(
        tokenizer=tokenizer,
        base_tokenizer=base_tokenizer,
        num_annotation_tokens=num_annotation_tokens,
        annotation_mask_probability=args.ann_mask,
        context_mask_probability=args.context_mask,
        sequence_mask_std=args.sequence_std,
    )

    kwargs = {
        'eval_strategy': 'steps',
        'eval_steps':args.save_steps,
        'save_strategy': 'steps',
        'save_steps':args.save_steps,
        'push_to_hub':True,
        'hub_strategy':'all_checkpoints',
        'hub_model_id':args.save_path,
        'hub_private_repo':True,
        'hub_always_push':True,
        'save_only_model':True,
        'warmup_steps':1000,
        #'include_inputs_for_metrics': True,
        #'use_cpu':True
    }

    if args.context and args.target:
        compute_metrics = compute_metrics_campv2_full
    elif args.context:
        compute_metrics = compute_metrics_campv2_gen
    elif args.target:
        compute_metrics = compute_metrics_campv2_rep
    else:
        print('Must specify --context or --target')

    args.grad_accum = 1
    trainer = hf_trainer_from_args(args,
                                   model=model,
                                   train_dataset=train_dataset,
                                   valid_dataset=valid_dataset,
                                   compute_metrics=compute_metrics,
                                   data_collator=data_collator, **kwargs)
    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    print(f'Random weight metrics: {metrics}')
    trainer.train()
    metrics = trainer.evaluate()
    print(f'Final metrics: {metrics}')
    trainer.model.push_to_hub(args.save_path, token=args.token, private=True)
    

if __name__ == '__main__':
    main()
