import torch
import argparse
from torchinfo import summary
from datasets import load_dataset
from transformers import EsmTokenizer
from udev_public.models.esm.custom_esm import CustomEsmForMaskedLM
from udev_public.models.bert.custom_bert import CustomBertForMaskedLM
from udev_public.data.data_collators import AspectEvaluationCollator
from udev_public.data.nlp_dataset_classes import SequenceAnnotationDataset
from udev_public.metrics.language_modeling import compute_metrics_mlm_from_pred
from udev_public.trainers.huggingface import hf_trainer_from_args
from udev_public.data.data_utils import get_max_from_list_of_lists
from udev_public.utils.logging import log_results


def get_args():
    parser = argparse.ArgumentParser(description='Settings')
    parser.add_argument('--token', type=str)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--tokenizer_path', type=str, default='facebook/esm2_t6_8m_UR50D')
    parser.add_argument('--at_path', type=str, default='GleghornLab/AT_EXP')
    parser.add_argument('--asm35_path', type=str, default='GleghornLab/asm35_exp')
    parser.add_argument('--data_path', type=str, default='GleghornLab/EXP_annotations')
    parser.add_argument('--log_path', type=str, default='EXP_fill.txt')
    return parser.parse_args()

def main():
    args = get_args()
    
    models = {
        'at_exp': {'path': args.at_path, 'type': 'AT'},
        'asm35_exp': {'path': args.asm35_path, 'type': 'ASM'}
    }

    aspect_ranges = {
        'EC': (1, 4449),
        'MF': (4451, 10708),
        'BP': (10710, 23298),
        'CC': (23300, 25169),
        'IP': (25171, 30607),
        '3D': (30609, 33234),
        'CO': (33236, 33324)
    }

    dataset = load_dataset(args.data_path, token=args.token)
    tokenizer = EsmTokenizer.from_pretrained(args.tokenizer_path)
    compute_metrics = compute_metrics_mlm_from_pred

    for model_name, model_info in models.items():
        print(f"\nEvaluating model: {model_name}")
        
        if model_info['type'] == 'AT':
            model = CustomBertForMaskedLM.from_pretrained(model_info['path'], token = args.token)
            both_values, AT = [False], True
        else:  # ASM
            model = CustomEsmForMaskedLM.from_pretrained(model_info['path'], token = args.token)
            both_values, AT = [False, True], False

        summary(model)

        for both in both_values:
            print(f"\nRunning evaluation with both={both}")
            for aspect in aspect_ranges.keys():
                print(f"\nEvaluating {aspect} aspect")
                test_set = dataset[f'test_{aspect.lower()}']
                plm_vocab_size = tokenizer.vocab_size
                max_ann_value = get_max_from_list_of_lists(test_set['combined'])
                
                if both:
                    vocab_size = plm_vocab_size + max_ann_value
                    print(f'Total vocab size: {vocab_size} (PLM: {plm_vocab_size}, Annotations: {max_ann_value})')
                else:
                    vocab_size = max_ann_value + 3  # +3 for special tokens
                    print(f'Annotation vocab size: {vocab_size}')
                
                eval_dataset = SequenceAnnotationDataset(test_set, 'seqs', 'combined', max_length=args.max_length)
                data_collator = AspectEvaluationCollator(
                    aspect_ranges=aspect_ranges,
                    aspect=aspect,
                    tokenizer=tokenizer,
                    both=both,
                    AT=AT
                )
                
                args.save_path, args.grad_accum, args.lr, args.num_epochs = 'mask_fill', 1, 1e-4, 1 # for get trainer
                trainer = hf_trainer_from_args(args,
                                               model,
                                               train_dataset=eval_dataset,
                                               valid_dataset=eval_dataset,
                                               compute_metrics=compute_metrics,
                                               data_collator=data_collator
                                               )
                
                metrics = trainer.evaluate()
                print(f"{aspect} Evaluation Metrics (both={both}):")
                print(metrics)
                
                log_results(args, metrics, f'{model_name}_{aspect}_{both}')
                trainer.accelerator.free_memory()
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
