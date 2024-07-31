import argparse
import torchinfo
from functools import partial
from datasets import load_dataset
from udev_public.data.data_collators import AnnotationCollator
from udev_public.data.nlp_dataset_classes import TokenizedDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm_from_pred
from udev_public.data.data_utils import get_max_from_list_of_lists, process_column
from udev_public.trainers.huggingface import hf_trainer_from_args
from udev_public.models.esm.custom_esm import CustomEsmForMaskedLM


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, default='lhallee/AT_new')
    parser.add_argument('--save_path', type=str, default='lhallee/AT_new')
    parser.add_argument('--data_path', type=str, default='lhallee/new_annotation_vocab')
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
    train_set = dataset['train']
    test_set = dataset['test']
    train_set = TokenizedDataset(train_set, col_name=col_name)
    test_set = TokenizedDataset(test_set, col_name=col_name)

    model = CustomEsmForMaskedLM.from_pretrained(args.model_path, token=args.token)
    print(torchinfo.summary(model))

    data_collator = AnnotationCollator(vocab_size, mlm_probability=0.15)
    args.num_epochs, args.grad_accum = 5, 1
    kwargs = {
        'eval_strategy': 'steps',
        'eval_steps':1000,
        'save_strategy': 'steps',
        'save_steps':5000,
        'push_to_hub':True,
        'hub_strategy':'all_checkpoints',
        'hub_model_id':args.save_path,
        'hub_private_repo':True,
        'hub_always_push':True,
        'save_only_model':True,
        'warmup_steps':100,
    }
    trainer = hf_trainer_from_args(args,
                                   model,
                                   train_set, 
                                   test_set,
                                   compute_metrics_mlm_from_pred,
                                   data_collator, **kwargs)

    metrics = trainer.evaluate()
    print(metrics)
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    trainer.model.push_to_hub(args.save_path, token=args.token, private=True)


if __name__ == '__main__':
    main()
