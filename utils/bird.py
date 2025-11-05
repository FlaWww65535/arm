from sql_metadata import Parser
from typing import List
from tiger_utils import read_json, write_json, read_pickle

def sql_to_tables(sql: str, db_id: str) -> List[str]:
  gold_ts = Parser(sql).tables
  gold_ts = [f'{db_id}#sep#{gold_t}' for gold_t in gold_ts]
  return gold_ts

def convert_db_to_tables(mode: str):
  databases = read_json(f'./data/bird/{mode}_databases.json')
  tables = {}

  for db in databases:
    db_id = db['db_id']
    for table_idx, table in enumerate(db['table_names_original']):
      tables[f'{db_id}#sep#{table}'] = {'db_id':db_id, 'table_name_original': table, 'table_name': db['table_names'][table_idx], 'column_names_original': [], 'column_names': [], 'column_types': []}

    for col_original, col, col_type in zip(db['column_names_original'][1:], db['column_names'][1:], db['column_types']):
      col_idx, col_original_name = col_original
      _, col_name = col
      tables[f"{db_id}#sep#{db['table_names_original'][col_idx]}"]['column_names_original'].append(col_original_name)
      tables[f"{db_id}#sep#{db['table_names_original'][col_idx]}"]['column_names'].append(col_name)
      tables[f"{db_id}#sep#{db['table_names_original'][col_idx]}"]['column_types'].append(col_type)

  write_json(tables, f'./data/bird/{mode}_tables.json')

def check_duplicates():
  tables = read_json('./data/bird/dev_tables.json')
  
  table_names_set = set()
  
  for table in tables:
    table_names_set.add(table.split('#sep#')[1])
  
  assert len(table_names_set) == len(tables)

  print(len(table_names_set), len(tables))

def get_db_for_table(t_name: str):
  tables = read_json('./data/bird/dev_tables.json')
  for t in tables:
    if t_name.upper() == tables[t]['table_name_original'].upper():
      return f"{tables[t]['db_id']}#sep#{tables[t]['table_name_original']}"

def train_needed_tables():
  train_needed_tables = set()
  for example in read_json('./react_examples.json')['bird']:
    for step in example['reasoning']:
      if type(step) is list:
        train_needed_tables.update(step[1:-1])

  for example in read_json('./data/bird/ete_examples.json'):
    train_needed_tables.update(example['candidate objects'])
  
  for example in read_json('./data/bird/retrieval_examples.json'):
    train_needed_tables.update(example['candidate objects'])

  return train_needed_tables

def get_corpus_rows(mode: str):
  if mode == 'train':
    tables = train_needed_tables()
  else:
    tables = read_json(f'./data/bird/{mode}_tables.json')

  rows_dict, rows_embeds_dict = {}, {}

  for t in tables:
    rows = read_pickle(f'./data/bird/{mode}_rows_original/{t}.pkl')
    rows_dict[t] = rows
  
  return rows_dict, rows_embeds_dict

def load_table_scores():
  dataset = 'bird'
  uniqueness = read_json(f'./data/{dataset}/dev_uniqueness.json')
  semantic_col_sim = read_json(f'./data/{dataset}/semantic_col_sim.json')
  exact_col_sim = read_json(f'./data/{dataset}/exact_col_sim.json')
  jaccard = read_json(f'./data/{dataset}/dev_jaccard.json')
  
  table_scores = {
    'jaccard': jaccard,
    'uniqueness': uniqueness,
    'semantic_col_sim': semantic_col_sim,
    'exact_col_sim': exact_col_sim
  }

  return table_scores

if __name__ == '__main__':
  convert_db_to_tables('train')
  # check_duplicates()
  # generate_vocabs('dev')

  # print(get_db_for_table('FRPM'))