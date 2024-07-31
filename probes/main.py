import argparse
import torch
import warnings
warnings.filterwarnings('ignore')
from udev_public.data.load_downstream_data import load_data_for_probe
from train import train

"""
TO USE
py -m main --plm_paths plm_id_like_to_test another_plm_to_test --data_paths the_datasets_id_like_to_test

Each plm_path needs an input_dim (hidden_dim of the model) and a model_type (ESM, ankh-base, etc.)

For classification
These datasets (just leave data_paths emtpy)
'GleghornLab/EC_reg', 'GleghornLab/CC_reg', 'GleghornLab/MF_reg', 'GleghornLab/BP_reg', 'GleghornLab/MetalIonBinding_reg', 'GleghornLab/dl_binary_reg', 'GleghornLab/dl_ten_reg'
use --hidden_dim 8192 (default), --num_layers 2 (default), --patience 10 (20 if you wanna be very careful)

For structure tasks
These datasets
'GleghornLab/ssq3', 'GleghornLab/ssq8'
use --hidden_dim 384, --intermediate_dim 1024 (default), --num_layers 1, --full, --patience 5, --read_scaler 100
"""


def main():
    parser = argparse.ArgumentParser(description='Probe PLMs on downstream tasks')
    # Paths
    parser.add_argument('--plm_paths', nargs='+', type=str, help='Paths to base models')
    parser.add_argument('--data_paths', nargs='+', type=str, help='Paths to data files')
    parser.add_argument('--save_path', nargs='+', type=str, help='Where to save hybrid model')
    parser.add_argument('--log_path', type=str, default='./results.txt', help='Path to log file (default: ./results.txt)')
    parser.add_argument('--output_dir', type=str, default='./trainer_output', help='Output directory (default: ./trainer_output)')
    parser.add_argument('--db_path', type=str, default='embeddings.db', help='Path to embeddings database (default: embeddings.db)')
    parser.add_argument('--subfolder', type=str, default='')
    # PLM
    parser.add_argument('--model_types', nargs='+', type=str, help='Choose from ESM, ANKH, CAMP, Protvec, Aspectvec')
    parser.add_argument('--aspects', nargs='+', type=str, help='Choose from EC, MF, BP, CC, IP, 3D, ALL')
    # Probe settings
    parser.add_argument('--input_dims', nargs='+', type=int, help='Input dimensions')
    parser.add_argument('--hidden_dim', default=8192, type=int, help='Hidden dimension (default: 8192)')
    parser.add_argument('--intermediate_dim', default=1024, type=int, help='Intermediate dimension (default: 1024)')
    parser.add_argument('--pooling', type=str, default='cls', help='Pooling method (default: cls)')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate (default: 0.1)')
    parser.add_argument('--num_layers', default=2, type=int, help='Number of layers (default: 2)')
    parser.add_argument('--nhead', default=8, type=int, help='Number of attention heads (default: 8)')
    # Data settings
    parser.add_argument('--trim', action='store_true', help='Trim sequences by max length')
    parser.add_argument('--random', action='store_true', help='Benchmark random vectors')
    parser.add_argument('--max_length', default=512, type=int, help='Maximum sequence length (default: 512)')
    parser.add_argument('--ppi', action='store_true', help='Use PPI dataset')
    parser.add_argument('--splits', nargs='+', help='Splits for datasets if getting from huggingface')
    # Training settings
    parser.add_argument('--full', action='store_true', help='Use matrix embedding')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size (default: 64)')
    parser.add_argument('--grad_accum', default=1, type=int, help='Gradient accumulation steps (default: 1)')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay (default: 0.01)')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs (default: 200)')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping (default: 20)')
    parser.add_argument('--seed', default=7, type=int, help='Random seed (default: 7)')
    parser.add_argument('--hybrid_finetune', action='store_true', help='Use hybrid fine-tuning')
    parser.add_argument('--steps', action='store_true', help='Use stepwise evaluation or epoch')
    # Other
    parser.add_argument('--not_sql', action='store_true', help='Store embeddings locally and stream')
    parser.add_argument('--HF', action='store_true', help='Get embeddings from huggingface')
    parser.add_argument('--read_scaler', type=int, default=1000, help='Read scaler (default: 1000)')
    parser.add_argument('--logging_steps', type=int, default=100, help='Logging steps (default: 100)')
    parser.add_argument('--save', action='store_true', help='Save model weights')
    parser.add_argument('--token', type=str, help='Token for authentication')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.plm_paths == None: ### Leave plm_paths emtpy to test all models listed below
        args.plm_paths = [
            "GleghornLab/esm3",
            "GleghornLab/asm35_red",
            "GleghornLab/asm35_exp",
        ]
        args.model_types = ['esm3', 'esm', 'esm']
        args.input_dims = [1536, 480, 480]
        args.splits = ['esm3', 'asm35_red', 'asm35_nat']

    if args.ppi:
        if args.data_paths == None:
            args.data_paths = ['GleghornLab/pinui_yeast_set']
        args.input_dims = [dim * 2 for dim in args.input_dims]
        args.titles = ['PPI']
        args.aspects = ['BP']

    elif args.data_paths == None:
        args.data_paths = [
            'GleghornLab/EC_reg',
            'GleghornLab/CC_reg',
            'GleghornLab/MF_reg',
            'GleghornLab/BP_reg',
            'GleghornLab/MetalIonBinding_reg',
            'GleghornLab/dl_binary_reg',
            'GleghornLab/dl_ten_reg'
        ]
        args.titles = ['EC', 'CC', 'MF', 'BP', 'MB', 'DL2', 'DL10']
        args.aspects = ['EC', 'CC', 'MF', 'BP', 'IP', 'CC', 'CC']

    print('\n-----Config-----\n')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    if args.model_types[0].lower() == 'aspectvec':
        data_paths = args.data_paths # we get all datasets
        plm_path = args.plm_paths[0] # we need the real path
        args.plm_paths = [plm_path] * len(data_paths) # we will technically run len(data_path) probes
        args.idx = list(range(len(args.plm_paths)))
        for data_path, idx in zip(data_paths, args.idx):
            args.data_paths = [data_path] # get single dataset
            data_tuple = load_data_for_probe(args)
            run_name = f'{args.aspects[idx]}_aspect_' + plm_path.split('/')[-1]
            args.db_path = './embeddings/' + run_name + '.db' # no full for aspect vec
            args.log_path = f'./results/{run_name}.txt'
            if not args.eval:
                train(args, idx, data_tuple)
            else:
                eval(args, idx, data_tuple)

    else:
        args.idx = list(range(len(args.plm_paths)))
        data_tuple = load_data_for_probe(args)
        for plm_path, model_type, idx in zip(args.plm_paths, args.model_types, args.idx):
            run_name = f'{model_type}_{plm_path.split("/")[-1]}'
            args.db_path = f'./embeddings/{args.full}_{run_name}.db'
            args.log_path = f'./results/{run_name}.txt'
            train(args, idx, data_tuple)


if __name__ == "__main__":
    main()
