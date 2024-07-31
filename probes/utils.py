import csv
from glob import glob


def main_result_to_csv(results_folder):
    results = glob(results_folder + '/*.txt')
    header = ['Model', 'EC', 'CC', 'MF', 'BP', 'MB', 'DL2', 'DL10', 'YPPI', 'HPPI', 'TS', 'SS3', 'SS8']
    csv_data = []
    for result in results:
        model = result.split('\\')[-1][:-4]
        print(model)
        row = [model] + ['-'] * 12
        with open(result, 'r') as f:
            for line in f:
                if 'data' in line.lower():
                    task = line.split('/')[-1].strip()
                    if task == 'EC_reg':
                        index = 1
                    elif task == 'CC_reg':
                        index = 2
                    elif task == 'MF_reg':
                        index = 3
                    elif task == 'BP_reg':
                        index = 4
                    elif task == 'MetalIonBinding_reg':
                        index = 5
                    elif task == 'dl_binary_reg':
                        index = 6
                    elif task == 'dl_ten_reg':
                        index = 7
                    elif task == 'pinui_yeast_set':
                        index = 8
                    elif task == 'pinui_human_set':
                        index = 9
                    elif task == 'Thermostability_reg':
                        index = 10
                    elif task == 'ssq3':
                        index = 11
                    elif task == 'ssq8':
                        index = 12
                if 'f1' in line.lower() or 'spearman' in line.lower():
                    f1 = float(line.split(':')[-1].strip())
                    row[index] = f1
        csv_data.append(row)
    csv_filename = 'main_results_.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)
    print(f"Results written to {csv_filename}")


def all_results_to_csv(results_folder):
    results = glob(results_folder + '/*.txt')
    header = ['Model']
    ts = ['TS_spearman', 'TS_r_sq']
    datasets = ['EC', 'CC', 'MF', 'BP', 'DL2', 'MB', 'DL10', 'YPPI', 'HPPI', 'SS3', 'SS8']
    metrics = ['f1', 'rec', 'prec']
    for dataset in datasets:
        for metric in metrics:
            header.append(f'{dataset}_{metric}')
    header.extend(ts)
    
    csv_data = []
    for result in results:
        model = result.split('\\')[-1][:-4]
        row = [model] + ['-'] * (len(header) - 1)
        with open(result, 'r') as f:
            current_dataset = ''
            for line in f:
                if 'data' in line.lower():
                    current_dataset = line.split('/')[-1].strip()
                    if current_dataset == 'EC_reg':
                        current_dataset = 'EC'
                    elif current_dataset == 'CC_reg':
                        current_dataset = 'CC'
                    elif current_dataset == 'MF_reg':
                        current_dataset = 'MF'
                    elif current_dataset == 'BP_reg':
                        current_dataset = 'BP'
                    elif current_dataset == 'MetalIonBinding_reg':
                        current_dataset = 'MB'
                    elif current_dataset == 'dl_binary_reg':
                        current_dataset = 'DL2'
                    elif current_dataset == 'dl_ten_reg':
                        current_dataset = 'DL10'
                    elif current_dataset == 'pinui_yeast_set':
                        current_dataset = 'YPPI'
                    elif current_dataset == 'pinui_human_set':
                        current_dataset = 'HPPI'
                    elif current_dataset == 'ssq3':
                        current_dataset = 'SS3'
                    elif current_dataset == 'ssq8':
                        current_dataset = 'SS8'
                    elif current_dataset == 'Thermostability_reg':
                        current_dataset = 'TS'
                
                for metric in metrics + ['spearman', 'r_sq']:
                    if metric in line.lower():
                        value = float(line.split(':')[-1].strip())
                        if current_dataset == 'TS':
                            if metric == 'spearman':
                                row[header.index('TS_spearman')] = value
                            elif metric == 'r_sq':
                                row[header.index('TS_r_sq')] = value
                        else:
                            column = f'{current_dataset}_{metric}'
                            if column in header:
                                row[header.index(column)] = value
        
        csv_data.append(row)
    
    csv_filename = 'all_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)
    print(f"All results written to {csv_filename}")


def loss_to_csv(results_folder):
    import csv
    from glob import glob
    results = glob(results_folder + '/*.txt')
    header = ['Model', 'EC', 'CC', 'MF', 'BP', 'MB', 'DL2', 'DL10', 'YPPI', 'HPPI', 'TS', 'SS3', 'SS8']
    csv_data = []
    for result in results:
        model = result.split('\\')[-1][:-4]
        print(model)
        row = [model] + ['-'] * 12
        with open(result, 'r') as f:
            for line in f:
                if 'data' in line.lower():
                    task = line.split('/')[-1].strip()
                    if task == 'EC_reg':
                        index = 1
                    elif task == 'CC_reg':
                        index = 2
                    elif task == 'MF_reg':
                        index = 3
                    elif task == 'BP_reg':
                        index = 4
                    elif task == 'MetalIonBinding_reg':
                        index = 5
                    elif task == 'dl_binary_reg':
                        index = 6
                    elif task == 'dl_ten_reg':
                        index = 7
                    elif task == 'pinui_yeast_set':
                        index = 8
                    elif task == 'pinui_human_set':
                        index = 9
                    elif task == 'Thermostability_reg':
                        index = 10
                    elif task == 'ssq3':
                        index = 11
                    elif task == 'ssq8':
                        index = 12
                if 'eval_loss' in line.lower() or 'test_loss' in line.lower():
                    f1 = float(line.split(':')[-1].strip())
                    row[index] = f1
        csv_data.append(row)
    csv_filename = 'loss_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(csv_data)
    print(f"Results written to {csv_filename}")


if __name__ == '__main__':
    path = './results/ann_paper/'
    #all_results_to_csv(path)
    main_result_to_csv(path)
    #loss_to_csv(path)
