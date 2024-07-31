import argparse
from torchinfo import summary
from transformers import AutoTokenizer
from datasets import load_dataset
from functools import partial
from udev_public.models.camp.modeling_camp import CAMP
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
    parser.add_argument('--nlp_path', type=str, default='lhallee/annotation_transformer_uniref90')
    parser.add_argument('--data_path', type=str, default='lhallee/annotations_uniref90')
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--ann', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = CAMPConfig(plm_path=args.plm_path,
                        nlp_path=args.nlp_path,
                        annotation_transformer=args.ann,
                        num_hidden_layers=args.num_hidden_layers,
                        token=args.token)
    model = CAMP(config)
    model.config.token = None # remove HF token
    model.freeze()
    summary(model)

    data = load_dataset(args.data_path, token=args.token)

    if args.ann:
        seq_col = 'seqs'
        nlp_col = 'combined'
        try:
            process = partial(process_column, col_name=nlp_col)    
            data = data.map(process)
        except:
            pass
        nlp_tokenizer = None
        num_annotations = model.nlp.config.vocab_size
    else:
        seq_col = 'seq'
        nlp_col = 'text'
        nlp_tokenizer = AutoTokenizer.from_pretrained(args.nlp_path, token=args.token)
        num_annotations = 0

    try:
        train = data['train']
        valid = data['test']
    except:
        data = data['train'].shuffle(seed=42).train_test_split(test_size=500)
        train = data['train']
        valid = data['test']

    train_dataset = CampDataset(train, seq_col, nlp_col)
    valid_dataset = CampDataset(valid, seq_col, nlp_col)
    plm_tokenizer = AutoTokenizer.from_pretrained(args.plm_path, token=args.token)
    data_collator = CampCollator(
        plm_tokenizer=plm_tokenizer,
        nlp_tokenizer=nlp_tokenizer,
        max_length_plm=2048,
        max_length_nlp=1024 if args.ann else 512,
        num_annotations=num_annotations,
        annotation_transformer=args.ann,
        masking=False
    )

    args.num_epochs, args.grad_accum = 1, 1

    kwargs = {
        'eval_strategy': 'steps',
        'eval_steps':1000,
        'save_strategy': 'steps',
        'save_steps':1000,
        'push_to_hub':True,
        'hub_strategy':'all_checkpoints',
        'hub_model_id':args.save_path,
        'hub_private_repo':True,
        'hub_always_push':True,
        'save_only_model':True,
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
