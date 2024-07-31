import torch
import argparse
import sqlite3
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from udev_public.data.embed_data import embed_dataset
from udev_public.models.embedding_models import get_plm


# Default values (only used when nothing is passed)
DEFAULT_MODEL_PATHS = {
    "esm2_8": "facebook/esm2_t6_8M_UR50D",
    "esm2_35": "facebook/esm2_t12_35M_UR50D",
    "esm2_150": "facebook/esm2_t30_150M_UR50D",
    "esm2_650": "facebook/esm2_t33_650M_UR50D",
    "ankh_base": "ankh-base",
    "ankh_large": "ankh-large",
    "protvec": "lhallee/ProteinVec",
    "camp_nat": "lhallee/camp_nlp_space",
    "camp_red": "lhallee/camp_ann_space",
    "camp_exp": "lhallee/CAMP_650_orig",
    "esm3": "GleghornLab/esm3",
    "random_esm": "facebook/esm2_t12_35M_UR50D",  # You can use any ESM model path here
    "random": "facebook/esm2_t12_35M_UR50D"
}

DEFAULT_MODEL_TYPES = {key: "ESM" for key in DEFAULT_MODEL_PATHS.keys()}
DEFAULT_MODEL_TYPES.update({
    "ankh_base": "ANKH",
    "ankh_large": "ANKH",
    "protvec": "protvec",
    "camp_nat": "camp",
    "camp_red": "camp",
    "camp_exp": "camp",
    "random_esm": "random_esm",
    "random": "random"
})

DEFAULT_DATA_PATHS = [
    'lhallee/EC_reg', 'lhallee/CC_reg', 'lhallee/MF_reg', 'lhallee/BP_reg',
    'lhallee/MetalIonBinding_reg', 'lhallee/dl_binary_reg', 'lhallee/dl_ten_reg',
    'lhallee/CLEAN', 'lhallee/pinui_yeast_set', 'lhallee/pinui_human_set'
]

def get_args():
    parser = argparse.ArgumentParser(description='Protein Language Model Embedding Generator')
    
    # Main arguments
    parser.add_argument('--token', type=str, help='Hugging Face token for dataset access')
    parser.add_argument('--action', choices=['sql', 'make'], required=True, 
                        help='Action to perform: "sql" to store embeddings in SQL, "make" to generate embeddings')
    parser.add_argument('--max_length', type=int, default=2048, 
                        help='Maximum sequence length for embedding (default: 2048)')
    
    # Model selection
    parser.add_argument('--model_names', nargs='+', default=[],
                        help='Names (split) for models to use')
    parser.add_argument('--model_paths', nargs='+', default=[],
                        help='Paths for models to use')
    parser.add_argument('--model_types', nargs='+', default=[],
                        help='Types for models to use')
    
    # Dataset selection
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATA_PATHS,
                        help='Datasets to process (default: all)')
    
    # Additional options
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (default: cuda if available, else cpu)')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save output files (default: ./output)')
    parser.add_argument('--HF_path', type=str, default='lhallee/plm_embeddings')
    
    return parser.parse_args()


def construct_model_dicts(args):
    if not args.model_names or not args.model_paths or not args.model_types:
        return DEFAULT_MODEL_PATHS, DEFAULT_MODEL_TYPES
    
    if len(args.model_names) != len(args.model_paths) or len(args.model_names) != len(args.model_types):
        raise ValueError("The number of model names, paths, and types must be equal")
    
    model_paths = dict(zip(args.model_names, args.model_paths))
    model_types = dict(zip(args.model_names, args.model_types))
    
    return model_paths, model_types


def make(args):
    args.full = False
    model_paths, model_types = construct_model_dicts(args)
    device = torch.device(args.device)

    total_seqs = []
    for path in tqdm(args.datasets, desc='Gathering Sequences'):
        dataset = load_dataset(path, token=args.token)
        for split, data in dataset.items():
            for column in data.column_names:
                if 'seq' in column.lower():
                    total_seqs.extend(data[column])

    seqs = list(set(total_seqs))
    seqs = [seq[:args.max_length] for seq in tqdm(seqs, desc=f'Trimming to max_length {args.max_length}')]
    print(f'Total unique sequences: {len(seqs)}')

    for model_name, plm_path in model_paths.items():
        model_type = model_types[model_name]
        plm = get_plm(args, plm_path, model_type, eval=True)
        if model_name != 'random':
            plm = plm.to(device)
        embeds = embed_dataset(args, plm, seqs)
        embedded_dataset = Dataset.from_dict({
            'seqs': seqs,
            'vectors': embeds,
        })
        embedded_dataset.push_to_hub(args.HF_path, token=args.token, split=model_name)


def to_sql(args):
    model_paths, _ = construct_model_dicts(args)
    for model_name, plm_path in model_paths.items():
        db_file = f"{args.output_dir}/{model_name}_vector_embeds.db"
        embedded_dataset = load_dataset(args.HF_path, token=args.token, split=model_name)
        seqs = embedded_dataset['seqs']
        vectors = embedded_dataset['vectors']
        
        with sqlite3.connect(db_file) as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS embeddings (sequence TEXT PRIMARY KEY, embedding BLOB)")
            for seq, vec in tqdm(zip(seqs, vectors), total=len(seqs), desc=f'Writing {model_name} embeddings'):
                c.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (seq, vec.tobytes()))
            conn.commit()


if __name__ == '__main__':
    args = get_args()
    if args.action == 'sql':
        to_sql(args)
    elif args.action == 'make':
        make(args)

"""
User Guide for Protein Language Model Embedding Generator

This script generates embeddings for protein sequences using various language models
and can store these embeddings in a SQLite database. It now includes options for random
embeddings and ESM models with random weights.

Usage:
python script.py --action <action> --token <YOUR_HF_TOKEN> [options]

Required Arguments:
  --action {make,sql}   Action to perform:
                        'make': Generate embeddings
                        'sql': Store generated embeddings in SQLite database
  --token TOKEN         Your Hugging Face token for dataset access

Optional Arguments:
  --model_names NAME [NAME ...]
                        Names (split) for models to use
  --model_paths PATH [PATH ...]
                        Paths for models to use
  --model_types TYPE [TYPE ...]
                        Types for models to use
  
  --datasets DATASET [DATASET ...]
                        Datasets to process. Default: all available datasets
  
  --max_length MAX_LENGTH
                        Maximum sequence length for embedding. Default: 2048
  
  --device {cuda,cpu}   Device to use for computation. Default: cuda if available, else cpu
  
  --output_dir OUTPUT_DIR
                        Directory to save output files. Default: ./output

Examples:
1. Generate embeddings using custom models:
   python script.py --action make --token YOUR_HF_TOKEN --model_names esm2_650 custom_model --model_paths facebook/esm2_t33_650M_UR50D /path/to/custom/model --model_types ESM CustomType

2. Store previously generated embeddings in SQLite database:
   python script.py --action sql --token YOUR_HF_TOKEN --model_names esm2_650 custom_model --model_paths facebook/esm2_t33_650M_UR50D /path/to/custom/model --model_types ESM CustomType

3. Use default models (when no model arguments are provided):
   python script.py --action make --token YOUR_HF_TOKEN

4. Use random embeddings:
   python script.py --action make --token YOUR_HF_TOKEN --model_names random --model_paths random --model_types random

5. Use ESM model with random weights:
   python script.py --action make --token YOUR_HF_TOKEN --model_names random_esm --model_paths facebook/esm2_t33_650M_UR50D --model_types random_esm

Notes:
- Ensure you have the required libraries installed (torch, transformers, datasets, etc.)
- The script requires an active internet connection to download models and datasets
- For large datasets or models, ensure you have sufficient disk space and RAM
- GPU usage is recommended for faster processing, especially for larger models
- When specifying custom models, ensure that the number of model names, paths, and types are equal
- If no model arguments are provided, the script will use a predefined set of default models
- The 'random' option generates random embeddings without using any specific model architecture
- The 'random_esm' option uses the ESM architecture with random weights
"""