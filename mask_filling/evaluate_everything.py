import torch
from torch.nn import CrossEntropyLoss
from transformers import EsmForMaskedLM, EsmTokenizer
import biotite.sequence as biotite_sequence
import biotite.sequence.align as biotite_align
import numpy as np
import pandas as pd
import random
import argparse
import ast
from datasets import load_dataset
from tqdm.auto import tqdm
from collections import defaultdict
from generation_utils import (
    mask_tokens, fill_masks_with_amino_acids, AminoSequenceAligner, sequence_acc,
    analyze_group, analyze_csv, get_model_type, filter_by_sequence_length
)
import Levenshtein
from udev_public.models.camp.modeling_campv3 import *
from udev_public.models.esm.generate_esm import EsmGenerator
from udev_public.models.camp.config_campv3 import CAMPv3Config
from prompting import build_prompt


DATA_PATH = 'GleghornLab/AnnotationVocab'  ###changed path
MODEL_PATHS = [
    'facebook/esm2_t30_150M_UR50D',
    'GleghornLab/GSM'
]

MASKING = ['0.05', '0.15', '0.3', '0.3sec', '0.5', '0.7', '1.0']
NS = [10]
KS = [1, 3, 10]
PS = []
TEMPERATURES = [0.1]
NUMBER_SEQUENCES = 1000
MAX_LENGTH = 1536

def levenshtein_similarity(seq1, seq2):
    distance = Levenshtein.distance(seq1, seq2)
    max_length = max(len(seq1), len(seq2))
    similarity = 1 - (distance / max_length)
    return similarity

def combined_generation_result(
    annotation_input_ids,
    annotations_attention_mask,
    model,
    tokenizer, 
    seq, 
    template_ids,
    template_mask,
    labels, 
    get_model_type, 
    n, 
    samp_method,
    k, 
    p,  
    temp, 
    device,
    view=False
):
    if get_model_type in ['ESM']:
        with torch.no_grad():
            final_seq = model.generate(
                template_ids,
                template_mask,
                n=n,
                samp_method=samp_method,
                k=k if samp_method == "topk" else None,
                p=p if samp_method == "nuc" else None,
                temperature=temp,
                device=device,
                entropy=True,
                view=view,
            )
    elif get_model_type == 'GSM':
        with torch.no_grad():
            final_seq = model.generate(
                annotation_input_ids,
                annotations_attention_mask,
                template_ids,
                template_mask,
                n=n,
                samp_method=samp_method,
                k=k if samp_method == "topk" else None,
                p=p if samp_method == "nuc" else None,
                temperature=temp,
                device=device,
                entropy=True,
                view=view,
            )
    else:
        raise ValueError(f"Unknown model type: {get_model_type}")
    
    return final_seq, labels

def load_model(model_path, token=None):
    if 'gsm' in model_path.lower():
        config = CAMPv3Config.from_pretrained(model_path, token=token)
        config.token = token
        model = CAMPv3.from_pretrained(model_path, config=config, token=token)
        return model.eval()
    else:
        return EsmGenerator.from_pretrained(model_path).eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str)
    args = parser.parse_args()
    token = args.token

    aligner = AminoSequenceAligner()
    tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8m_UR50D')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset once
    data = load_dataset(DATA_PATH, split='train', token=token)    ###split
    data = data.filter(filter_by_sequence_length)
    data = data.shuffle(seed=42)
    data = data.select(range(NUMBER_SEQUENCES))
    seqs = data['seqs']
    gsm_annotations = [ast.literal_eval(ann) for ann in data['combined']]
    esm_annotations = [None] * len(seqs)

    for masking in tqdm(MASKING, desc="Masking Strategies"):
        if 'sec' in masking:
            sectional = True
            percentage = float(masking.replace('sec', ''))
        else:
            sectional = False
            percentage = float(masking)

        masked_sequences = []
        for seq in seqs:
            if percentage < 1.0:
                template = tokenizer(seq, add_special_tokens=True, return_tensors='pt')
                template_ids, template_attention_mask = template['input_ids'], template['attention_mask']
                labels = template_ids.clone()
                template_ids, labels = mask_tokens(template_ids, labels, percentage, tokenizer, sectional)
            else:
                original_ids = tokenizer(seq, add_special_tokens=True, return_tensors='pt')['input_ids']
                labels = original_ids.clone()
                template_ids = tokenizer('M', add_special_tokens=False)['input_ids']
                template_ids = [tokenizer.cls_token_id] + template_ids + [tokenizer.mask_token_id] * (len(seq)-1) + [tokenizer.eos_token_id]
                template_ids = torch.tensor(template_ids).unsqueeze(0)
                template_attention_mask = torch.ones_like(template_ids)

            masked_sequences.append((template_ids, template_attention_mask, labels))

        for model_path in tqdm(MODEL_PATHS, desc="Models"):
            model = load_model(model_path, args.token).to(device)
            current_model_type = get_model_type(model_path)

            results = []
            
            annotations = gsm_annotations if current_model_type in ['GSM'] else esm_annotations

            for seq, ann, (template_ids, template_attention_mask, labels) in tqdm(zip(seqs, annotations, masked_sequences), leave=False, desc='Main loop'):
                if current_model_type in ['GSM']:
                    annotation_input_ids, annotation_attention_mask = build_prompt(ann)
                else:
                    annotation_input_ids, annotation_attention_mask = None, None

                random_fill = fill_masks_with_amino_acids(template_ids, tokenizer)
                random_fill = model.decode_seq(random_fill[0])
                random_a_b, random_score = aligner.align(seq, random_fill)
                random_acc = sequence_acc(seq, random_fill)
                random_lev_sim = levenshtein_similarity(seq, random_fill)
                random_lev_dist = Levenshtein.distance(seq, random_fill)

                results.append({
                    'percentage': percentage,
                    'sectional': sectional,
                    'k': None,
                    'n': None,
                    'p': None,
                    'temperature': None,
                    'Random': True,
                    'a_b': random_a_b,
                    'score': random_score,
                    'accuracy': random_acc,
                    'levenshtein_similarity': random_lev_sim,
                    'levenshtein_distance': random_lev_dist,
                    'generated_seq': random_fill,
                    'original_seq': seq
                })

                for temp in tqdm(TEMPERATURES, desc="Temperatures", leave=False):
                    for n in tqdm(NS, desc="NS", leave=False):
                        for k in tqdm(KS, desc='KS', leave=False):
                            samp_method = "topk"
                            
                            final_seq, _ = combined_generation_result(
                                annotation_input_ids,
                                annotation_attention_mask,
                                model,
                                tokenizer, 
                                seq,
                                template_ids.clone(),
                                template_attention_mask,
                                labels,
                                current_model_type,
                                n,
                                samp_method,
                                k,
                                None, 
                                temp,
                                device
                            )
                            
                            real_a_b, real_score = aligner.align(seq, final_seq)
                            real_acc = sequence_acc(seq, final_seq)
                            real_lev_sim = levenshtein_similarity(seq, final_seq)
                            real_lev_dist = Levenshtein.distance(seq, final_seq)

                            results.append({
                                'percentage': percentage,
                                'sectional': sectional,
                                'k': k,
                                'n': n,
                                'p': None,
                                'temperature': temp,
                                'Random': False,
                                'a_b': real_a_b,
                                'score': real_score,
                                'accuracy': real_acc,
                                'levenshtein_similarity': real_lev_sim,
                                'levenshtein_distance': real_lev_dist,
                                'generated_seq': final_seq,
                                'original_seq': seq
                            })

                        for p in tqdm(PS, desc='PS', leave=False):
                            samp_method = "nuc"
                            
                            final_seq, _ = combined_generation_result(
                                annotation_input_ids,
                                annotation_attention_mask,
                                model,
                                tokenizer,
                                seq,
                                template_ids.clone(),
                                template_attention_mask,
                                labels,
                                current_model_type,
                                n,
                                samp_method,
                                None,
                                p,
                                temp,
                                device
                            )
                            
                            real_a_b, real_score = aligner.align(seq, final_seq)
                            real_acc = sequence_acc(seq, final_seq)
                            real_lev_sim = levenshtein_similarity(seq, final_seq)
                            real_lev_dist = Levenshtein.distance(seq, final_seq)

                            results.append({
                                'percentage': percentage,
                                'sectional': sectional,
                                'k': None,
                                'n': n,
                                'p': p,
                                'temperature': temp,
                                'Random': False,
                                'a_b': real_a_b,
                                'score': real_score,
                                'accuracy': real_acc,
                                'levenshtein_similarity': real_lev_sim,
                                'levenshtein_distance': real_lev_dist,
                                'generated_seq': final_seq,
                                'original_seq': seq
                            })
                        
            df = pd.DataFrame(results)
            model_name = model_path.split('/')[-1]
            df.to_csv(f'{model_name}_generation_results_raw_{percentage}{"_sec" if sectional else ""}.csv', index=False)

            del model
            torch.cuda.empty_cache()
