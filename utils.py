import os
import csv

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
      writer.writerow(['dataset', 'readout', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'avg', 'std'])
    writer.writerow(result)
    