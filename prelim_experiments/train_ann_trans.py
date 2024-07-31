import argparse
import torchinfo
from functools import partial
from datasets import load_dataset
from transformers import EsmConfig, EsmForMaskedLM 
from udev_public.data.data_collators import AnnotationCollator
from udev_public.data.nlp_dataset_classes import TokenizedDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm
from udev_public.data.data_utils import get_max_from_list_of_lists, process_column
from udev_public.trainers.huggingface import hf_trainer_from_args


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='lhallee/annotation_transformer_uniref90')
    parser.add_argument('--data_path', type=str, default='lhallee/annotations_uniref90')
    return parser.parse_args()


def main():
    args = get_args()

    dataset = load_dataset(args.data_path, token=args.token)
    split_name, col_name = 'unique_anns', 'combined'
    try:
        process = partial(process_column, col_name=col_name)
        dataset = dataset.map(process)
    except:
        pass
    vocab_size = get_max_from_list_of_lists(dataset[split_name][col_name])
    print('Vocab size ', vocab_size)

    dataset = dataset[split_name].shuffle(seed=42).train_test_split(test_size=1000, seed=42)
    train_set = dataset[split_name]
    test_set = dataset[split_name]
    train_set = TokenizedDataset(train_set, col_name=col_name)
    test_set = TokenizedDataset(test_set, col_name=col_name)
 
    config = EsmConfig()
    config.vocab_size = vocab_size + 4
    config.pad_token_id = 0
    config.mask_token_id = vocab_size + 3
    config.token_dropout = True
    config.hidden_size = 384
    config.intermediate_size = 2048
    config.position_embedding_type = 'rotary'
    config.max_position_embeddings = 1
    config.num_hidden_layers = 1
    config.num_attention_heads = 8

    model = EsmForMaskedLM(config)
    print(torchinfo.summary(model))

    data_collator = AnnotationCollator(vocab_size, mlm_probability=0.15)
    args.num_epochs = 10
    trainer = hf_trainer_from_args(args, model, train_set, test_set, compute_metrics_mlm, data_collator)

    metrics = trainer.evaluate()
    print(metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    trainer.model.push_to_hub(args.save_path, token=args.token, private=True)


if __name__ == '__main__':
    main()
