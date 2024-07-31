import argparse
import torch
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


"""
For trying different loss combinations with the original CAMP model
"""



def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--effective_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--plm_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--nlp_path', type=str, default='lhallee/annotation_transformer_uniref90')
    parser.add_argument('--data_path', type=str, default='lhallee/annotations_uniref90')
    parser.add_argument('--num_hidden_layers', type=int, default=1)
    parser.add_argument('--ann', action='store_true')
    parser.add_argument('--set', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    args.num_epochs, args.grad_accum = 1, 1

    seq_col = 'seqs'
    nlp_col = 'combined'
    process = partial(process_column, col_name=nlp_col)
    data = load_dataset(args.data_path, token=args.token)
    data = data.map(process)
    train = data['train'].shuffle(seed=42)
    train_dataset = CampDataset(train, seq_col, nlp_col)

    if args.set:
        diffs = [True]
        mnrs = [True]
        mlms = [True]
        latents = [False]
    else:
        diffs = [True, True, True]
        mnrs = [True, False, False]
        mlms = [False, True, False]
        latents = [False, False, True]

    for diff, mnr, mlm, latent in zip(diffs, mnrs, mlms, latents):
        config = CAMPConfig(plm_path=args.plm_path,
                            nlp_path=args.nlp_path,
                            annotation_transformer=args.ann,
                            num_hidden_layers=args.num_hidden_layers,
                            token=args.token)
        config.diff = diff
        config.mnr = mnr
        config.mlm = mlm
        config.latent = latent

        model = CAMP(config)
        model.config.token = None # remove HF token
        model.freeze()
        summary(model)

        plm_tokenizer = AutoTokenizer.from_pretrained(args.plm_path, token=args.token)
        if args.ann:
            nlp_tokenizer = None
            num_annoations = model.nlp.config.vocab_size
        else:
            nlp_tokenizer = AutoTokenizer.from_pretrained(args.nlp_path, token=args.token)
            num_annoations = 0

        data_collator = CampCollator(
            plm_tokenizer=plm_tokenizer,
            nlp_tokenizer=nlp_tokenizer,
            max_length_plm=2048,
            max_length_nlp=1024,
            num_annotations=num_annoations,
            annotation_transformer=args.ann,
            mlm_probability=0.05,
        )

        save_path = f'lhallee/camp8_ob_{diff}_{mnr}_{mlm}_{latent}'
        args.save_path = save_path
        trainer = hf_trainer_from_args(args, model, train_dataset, data_collator=data_collator)
        trainer.train()
        trainer.model.push_to_hub(save_path, token=args.token, private=True)
        trainer.accelerator.free_memory()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    