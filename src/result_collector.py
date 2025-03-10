import os
import re
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--exp_list", type=str, required=True, nargs='+')
args = parser.parse_args()

experiments_list = args.exp_list
dataset = args.dataset
print(f"Collect results from these experiments: {experiments_list}.")

def extract_best_results(log_dir, exp_list, output_csv, dataset):
    best_pattern = re.compile(f'█████████████ BEST ████████████████')
    metric_pattern = re.compile(r'(\w+@\d+):\s*([\d.]+)')
    headers = ['Log File',
               'Experiment',
               'recall@5',
               'recall@10',
               'recall@20',
               'ndcg@5',
               'ndcg@10',
               'ndcg@20',
               'precision@5',
               'precision@10',
               'precision@20',
               'map@5',
               'map@10',
               'map@20'
               ]
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers)

        for log_file in os.listdir(log_dir):
            if log_file.endswith('.log'):
                parts = log_file.split('-')
                experiment_name = parts[2]
                dataset_name = parts[1]

                if experiment_name in exp_list and dataset_name == dataset:
                    correct_file = False
                    log_path = os.path.join(log_dir, log_file)

                    best_exist = False
                    with open(log_path, 'r', encoding='utf-8') as f:
                        for line in f.readlines():
                            match = best_pattern.search(line)
                            if match:
                                best_exist = True
                                continue
                            if best_exist:
                                correct_prefix = line.split(' ')[0] == "Test:"
                                metric_matches = metric_pattern.findall(line)
                                if metric_matches and correct_prefix:
                                    metrics_dict = {metric: float(value) for metric, value in metric_matches}
                                    row = [log_file, experiment_name] + [metrics_dict.get(header, None) for header in headers[2:]]
                                    csv_writer.writerow(row)
                                    print(f"The results in file: {log_file} has been recored into {output_csv}.")
                                    correct_file = True
                                    break
                                else:
                                    pass
                    if not correct_file:
                        print(f"======> ERROR: The file: {log_file} isn't a correct log file, don't contain complete experiment logging.")
                else:
                    continue


log_directory = "./log"
output_csv_file = "./results.csv"
extract_best_results(log_directory, experiments_list, output_csv_file, dataset)