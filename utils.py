import os
import csv

import torch

def create_directory(directory_path):
  try:
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
  except OSError:
    print('Creating', directory_path, 'directory error')
      
def save_result_csv(file_path, result, print_column_name, mode='a'):
  with open(file_path, mode, newline='\n') as file:
    writer = csv.writer(file)
    if print_column_name:
      writer.writerow(['dataset', 'readout', 'fold 1', 'fold 2', 'fold 3', 'fold 4', 'fold 5', 'fold 6', 'fold 7', 'fold 8', 'fold 9', 'fold 10', 'mean of acc', 'std of acc', 'time per epochs'])
    writer.writerow(result)