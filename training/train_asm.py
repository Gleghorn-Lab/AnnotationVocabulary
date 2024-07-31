import torch
import argparse
from torchinfo import summary
from functools import partial
from datasets import load_dataset
from transformers import EsmTokenizer
from udev_public.models.esm.custom_esm import CustomEsmForMaskedLM
from udev_public.data.data_collators import SequenceAnnotationCollator
from udev_public.data.nlp_dataset_classes import SequenceAnnotationDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm_from_pred
from udev_public.data.data_utils import get_max_from_list_of_lists, process_column
from udev_public.trainers.huggingface import hf_trainer_from_args


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--esm_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--save_path', type=str, default='lhallee/test')
    parser.add_argument('--data_path', type=str, default='GleghornLab/EXP_annotations')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--columns', type=str, nargs='+')
    parser.add_argument('--new', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    args.grad_accum = 1
    dataset = load_dataset(args.data_path, token=args.token)
    if args.columns == None:
        seq_col, ann_col = 'seqs', 'combined'
    else:
        seq_col, ann_col = args.columns
    try:
        process = partial(process_column, col_name=ann_col)
        dataset = dataset.map(process)
    except:
        pass

    try:
        test_set = dataset['test']
        train_set = dataset['train']
    except:
        dataset = dataset['train'].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
        train_set = dataset['train']
        test_set = dataset['test']

    vocab_size = max(get_max_from_list_of_lists(dataset['train'][ann_col]), get_max_from_list_of_lists(dataset['test'][ann_col]))
    print('Vocab size ', vocab_size)
    train_dataset = SequenceAnnotationDataset(train_set, seq_col, ann_col, max_length=args.max_length)
    test_dataset = SequenceAnnotationDataset(test_set, seq_col, ann_col, max_length=args.max_length)

    tokenizer = EsmTokenizer.from_pretrained(args.esm_path)
    plm_vocab_size = tokenizer.vocab_size

    model = CustomEsmForMaskedLM.from_pretrained(args.esm_path)
    if args.new:
        with torch.no_grad():
            total_vocab = vocab_size + plm_vocab_size + 1
            model.resize_token_embeddings(total_vocab)
            model.lm_head.bias = torch.nn.Parameter(torch.ones(total_vocab))

    print(summary(model))

    kwargs = {
        'eval_strategy': 'steps',
        'eval_steps': 1000,
        'save_strategy': 'steps',
        'save_steps': 1000,
    }

    data_collator = SequenceAnnotationCollator(plm_tokenizer=tokenizer)
    trainer = hf_trainer_from_args(args,
                                   model,
                                   train_dataset=train_dataset,
                                   valid_dataset=test_dataset,
                                   compute_metrics=compute_metrics_mlm_from_pred,
                                   data_collator=data_collator, **kwargs)

    metrics = trainer.evaluate()
    print(metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    trainer.model.push_to_hub(args.save_path, token=args.token, private=True)

if __name__ == '__main__':
    main()
