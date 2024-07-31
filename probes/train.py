import os
import torch
from torchinfo import summary
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
from tqdm.auto import tqdm
from udev_public.models.probes import LinearProbe, BertProbe, ConvBertProbe
from udev_public.models.convbert.config_convbert import ConvBertConfig
from udev_public.models.embedding_models import HybridModel, get_plm
from udev_public.data.nlp_dataset_classes import (
    FineTuneDatasetEmbedsFromDisk,
    FineTuneDatasetEmbeds,
    PPIDatasetEmbedsFromDisk,
    PPIDatasetEmbeds,
    SequenceLabelDataset
)
from udev_public.data.embed_data import embed_dataset_and_save, embed_dataset
from udev_public.data.data_utils import read_sequences_from_db
from udev_public.data.load_downstream_data import get_data
from udev_public.data.data_collators import collate_fn_embeds, collate_seq_labels
from udev_public.metrics.classification import *
from udev_public.metrics.regression import *
from udev_public.utils.logging import log_results


def train(args, idx, data_tuple):
    all_seqs, train_sets, valid_sets, test_sets, num_labels, task_types = data_tuple
    datasets = []
    input_dim = args.input_dims[idx]
    aspect = args.aspects[idx] if args.aspects != None else None
    plm_path = args.plm_paths[idx]
    model_type = args.model_types[idx]
    emb_dict = None

    ### Dataset creation
    print('\n-----Embedding-----\n')
    if not args.not_sql: # if using sql, embed dataset to disk
        if os.path.exists(args.db_path):
            embedded_seqs = read_sequences_from_db(args.db_path)
            print('Embedding seqs: ', len(embedded_seqs))
            not_embedded_seqs = list(set(all_seqs) - set(embedded_seqs))
            if len(not_embedded_seqs) > 0:
                plm = get_plm(args, plm_path, model_type, aspect, subfolder=args.subfolder)
                plm = plm.to(args.device)
                embed_dataset_and_save(args, plm, not_embedded_seqs)
                plm.to('cpu')
                del plm
        else:
            plm = get_plm(args, plm_path, model_type, aspect, subfolder=args.subfolder)
            plm = plm.to(args.device)
            embed_dataset_and_save(args, plm, all_seqs)
            plm.to('cpu')
            del plm
    
        # use dataset from disk
        if args.ppi:
            for i in range(len(train_sets)):
                train_dataset = PPIDatasetEmbedsFromDisk(args, train_sets[i][0][0], train_sets[i][0][1], train_sets[i][1], input_dim//2)
                valid_dataset = PPIDatasetEmbedsFromDisk(args, valid_sets[i][0][0], valid_sets[i][0][1], valid_sets[i][1], input_dim//2)
                test_dataset = PPIDatasetEmbedsFromDisk(args, test_sets[i][0][0], test_sets[i][0][1], test_sets[i][1], input_dim//2)
                datasets.append((train_dataset, valid_dataset, test_dataset))

        else:
            for i in range(len(train_sets)):
                train_dataset = FineTuneDatasetEmbedsFromDisk(args, train_sets[i][0], train_sets[i][1], input_dim, task_types[i])
                valid_dataset = FineTuneDatasetEmbedsFromDisk(args, valid_sets[i][0], valid_sets[i][1], input_dim, task_types[i])
                test_dataset = FineTuneDatasetEmbedsFromDisk(args, test_sets[i][0], test_sets[i][1], input_dim, task_types[i])
                datasets.append((train_dataset, valid_dataset, test_dataset))

    else: # else, embed into memory or load from HF
        if args.HF:
            data = load_dataset('lhallee/plm_embeddings', token=args.token, split=args.splits[idx])
            seqs, vecs = data['seqs'], data['vectors']
            data_dict = dict(zip(seqs, vecs))
            embeddings = []
            for seq in tqdm(all_seqs, desc='Loading from HF'):
                embeddings.append(torch.tensor(data_dict[seq]))
            emb_dict = dict(zip(all_seqs, embeddings))
            del seqs
            del vecs
            del data_dict
            del data

        if args.ppi:
            if emb_dict is None:
                if args.random:
                    embeddings = [torch.rand(1, input_dim // 2) for _ in all_seqs] # input_dim is twice as big
                    emb_dict = dict(zip(all_seqs, embeddings))
                else:
                    plm = get_plm(args, plm_path, model_type, aspect, subfolder=args.subfolder)
                    plm = plm.to(args.device)
                    emb_dict = dict(zip(all_seqs, embed_dataset(args, plm, all_seqs)))
                    plm.to('cpu')
                    del plm
            for i in range(len(train_sets)):
                train_dataset = PPIDatasetEmbeds(args, emb_dict, train_sets[i][0][0], train_sets[i][0][1], train_sets[i][1])
                valid_dataset = PPIDatasetEmbeds(args, emb_dict, valid_sets[i][0][0], valid_sets[i][0][1], valid_sets[i][1])
                test_dataset = PPIDatasetEmbeds(args, emb_dict, test_sets[i][0][0], test_sets[i][0][1], test_sets[i][1])
                datasets.append((train_dataset, valid_dataset, test_dataset))

        else:
            if emb_dict is None:
                if args.random:
                    embeddings = [torch.rand(1, input_dim) for _ in all_seqs]
                    emb_dict = dict(zip(all_seqs, embeddings))
                else:
                    plm = get_plm(args, plm_path, model_type, aspect, subfolder=args.subfolder)
                    plm = plm.to(args.device)
                    emb_dict = dict(zip(all_seqs, embed_dataset(args, plm, all_seqs)))
                    plm.to('cpu')
                    del plm
            for i in range(len(train_sets)):
                train_dataset = FineTuneDatasetEmbeds(args, emb_dict, train_sets[i][0], train_sets[i][1], task_types[i])
                valid_dataset = FineTuneDatasetEmbeds(args, emb_dict, valid_sets[i][0], valid_sets[i][1], task_types[i])
                test_dataset = FineTuneDatasetEmbeds(args, emb_dict, test_sets[i][0], test_sets[i][1], task_types[i])
                datasets.append((train_dataset, valid_dataset, test_dataset))

    torch.cuda.empty_cache() # make sure cuda memory is freed

    ### Train probe
    print('\n-----Train-----\n')
    for i, dataset in enumerate(datasets):
        print(f'Training {args.data_paths[i]}, {i+1} / {len(datasets)}')
        train_dataset, valid_dataset, test_dataset = dataset
        task_type, num_label = task_types[i], num_labels[i]
        data_collator = collate_fn_embeds(full=args.full,
                                          max_length=args.max_length,
                                          task_type=task_type)

        ### For tokenwise tasks use bert
        ### For fixed length vectors use linear NN
        ### else use conv bert
        if args.full and task_type == 'tokenwise':
            model = BertProbe(args, input_dim=input_dim, task_type=task_type, num_labels=num_label)
        elif args.full and task_type != 'tokenwise':
            config = ConvBertConfig(
                input_size=input_dim,
                hidden_size=args.hidden_dim,
                intermediate_size=args.intermediate_dim,
                task_type=task_type,
                num_labels=num_label
            )
            print(config.num_labels)
            model = ConvBertProbe(config)
        else:
            model = LinearProbe(args, input_dim=input_dim, task_type=task_type, num_labels=num_label)
        summary(model)

        # get metrics function
        if task_type == 'singlelabel' or task_type == 'tokenwise':
            compute_metrics = compute_metrics_single_label_classification
        elif task_type == 'multilabel':
            compute_metrics = compute_metrics_multi_label_classification
        else:
            compute_metrics = compute_metrics_regression

        # use steps or epochs
        strategy_args = {
            'save_strategy': 'steps' if args.steps else 'epoch',
            'evaluation_strategy': 'steps' if args.steps else 'epoch',
            'logging_strategy': 'steps' if args.steps else 'epoch',
            'save_steps': 1000 if args.steps else None,
            'eval_steps': 1000 if args.steps else None,
            'logging_steps': 1000 if args.steps else None,
        }

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            seed=args.seed,
            data_seed=args.seed,
            optim='adamw_torch',
            load_best_model_at_end=True,
            fp16=args.fp16,
            save_total_limit=3,
            **strategy_args  # Include the strategy arguments
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )
        ### Evaluate on random weights, train, evaluate for a final time
        metrics = trainer.evaluate(test_dataset)
        print('Initial metrics\n', metrics)
        trainer.train()
        metrics = trainer.evaluate(test_dataset)
        print('Probe metrics\n', metrics)
        log_results(args, metrics, args.data_paths[i])

        probe = trainer.model.cpu() # save probe if going to use
        trainer.accelerator.free_memory() # make sure cuda memory is freed
        torch.cuda.empty_cache()

        if args.hybrid_finetune: # stick trained probe on top of full model
            print('\n-----Hybrid Fine-tuning-----\n')
            # Load the PLM again
            plm, tokenizer = get_plm(args, idx, eval=False, return_tokenizer=True, subfolder=args.subfolder)
            plm.train()
            # Create a combined model
            model = HybridModel(plm, probe, full=args.full)
            summary(model)

            # Create new datasets without pre-computed embeddings
            train_set, valid_set, test_set, num_label, task_type = get_data(args, args.data_paths[i])
            train_dataset = SequenceLabelDataset(train_set)
            valid_dataset = SequenceLabelDataset(valid_set)
            test_dataset = SequenceLabelDataset(test_set)
            data_collator = collate_seq_labels(tokenizer)

            scale = 10
            training_args = TrainingArguments(
                output_dir=args.output_dir + '_hybrid',
                num_train_epochs=5,
                learning_rate=args.lr / scale,
                weight_decay=args.weight_decay,
                per_device_train_batch_size=max(1, args.batch_size // scale), # model is much bigget
                per_device_eval_batch_size=max(1, args.batch_size // scale), # so we adjust batch size
                gradient_accumulation_steps=args.grad_accum * scale, # and gradient accumulation and lr
                seed=args.seed,
                data_seed=args.seed,
                optim='adamw_torch',
                load_best_model_at_end=True,
                fp16=args.fp16,
                save_total_limit=3,
                **strategy_args,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
            )
            
            metrics = trainer.evaluate(test_dataset)
            print('Added probe to full model\n', metrics)
            trainer.train()
            metrics = trainer.evaluate(test_dataset)
            print('Hybrid model metrics\n', metrics)
            log_results(args, metrics, args.data_paths[i])

            trainer.model.push_to_hub(args.save_path, token=args.token, private=True)
            tokenizer.push_to_hub(args.save_path, token=args.token, private=True)

            trainer.accelerator.free_memory() # make sure cuda memory is freed
            torch.cuda.empty_cache()
