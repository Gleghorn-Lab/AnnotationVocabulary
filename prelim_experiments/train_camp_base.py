import argparse
from torchinfo import summary
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from functools import partial
from udev_public.models.camp.modeling_camp import CAMPfinal
from udev_public.models.camp.config_camp import CAMPConfig
from udev_public.data.data_utils import process_column
from udev_public.data.nlp_dataset_classes import CampDataset
from udev_public.data.data_collators import CampCollator
from udev_public.trainers.huggingface import hf_trainer_from_args
from udev_public.metrics.similarity import compute_metrics_double


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='lhallee/CAMP_test')
    parser.add_argument('--plm_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--nlp_path', type=str, default='lhallee/AT_new')
    parser.add_argument('--data_path', type=str, default='lhallee/new_annotation_vocab')
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=1000)
    return parser.parse_args()


def main():
    args = get_args()
    config = CAMPConfig(plm_path=args.plm_path,
                        nlp_path=args.nlp_path,
                        annotation_transformer=True,
                        num_hidden_layers=args.num_hidden_layers,
                        token=args.token)
    model = CAMPfinal(config)
    model.config.token = None # remove HF token
    model.freeze()
    summary(model)

    data = load_dataset(args.data_path, token=args.token)
    seq_col, nlp_col = 'seqs', 'combined'
    try:
        process = partial(process_column, col_name=nlp_col)    
        data = data.map(process)
    except:
        pass
    nlp_tokenizer = None
    num_annotations = model.nlp.config.vocab_size
    train = concatenate_datasets([
        data['train'],
        data['exp'],
        data['long_exp']
    ])
    valid = data['test'].shuffle(seed=42).select(range(1000))

    train_dataset = CampDataset(train, seq_col, nlp_col)
    valid_dataset = CampDataset(valid, seq_col, nlp_col)
    data_collator = CampCollator(
        plm_tokenizer=model.tokenizer, # ankh tokenizer
        nlp_tokenizer=nlp_tokenizer,
        max_length_plm=2048,
        max_length_nlp=256,
        num_annotations=num_annotations,
        annotation_transformer=True,
        masking=False
    )

    args.num_epochs, args.grad_accum = 10, 1

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
        'warmup_steps':100,
        #'use_cpu':True
    }

    trainer = hf_trainer_from_args(args,
                                   model=model,
                                   train_dataset=train_dataset,
                                   valid_dataset=valid_dataset,
                                   compute_metrics=compute_metrics_double,
                                   data_collator=data_collator, **kwargs)
    metrics = trainer.evaluate(eval_dataset=valid_dataset)
    print(f'Random weight metrics: {metrics}')
    trainer.train()
    metrics = trainer.evaluate()
    print(f'Final metrics: {metrics}')
    trainer.model.push_to_hub(args.save_path, token=args.token, private=True)
    

if __name__ == '__main__':
    main()
