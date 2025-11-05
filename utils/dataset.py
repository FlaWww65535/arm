import json
from typing import Optional, List
import numpy as np
import pandas as pd

def create_table_statement(db_id:str, table_name: str, primary:bool = True):
  with open('./data/bird/schemas.json') as f:
    db = json.load(f)[db_id]
  table = [x for x in db if x['name'].lower() == table_name.lower()]
  assert len(table) == 1
  table = table[0]
  columns = table['columns']

  _schema_info, pkey_info, fkey_info = '', '', ''
  schema_info = f'CREATE TABLE `{table["name"]}` (\n'
  for i in range(len(columns)):
    _schema_info += f'  `{columns[i]["name"]}` {columns[i]["type"].upper()},\n'
    
    if primary and 'foreign_key' in columns[i]:
      fkey = columns[i]['foreign_key']
      fkey = f"`{fkey['table']}` (`{fkey['column']}`)"
      fkey_info += f'  FOREIGN KEY (`{columns[i]["name"]}`) REFERENCES {fkey},\n'

  additional_info = pkey_info + fkey_info
  if len(additional_info) > 0:
    schema_info += _schema_info + additional_info[:-2] # this removes the trailing ,\n of the last row
  else:
    schema_info += _schema_info[:-2]
  schema_info += '\n)\n\n'
  return schema_info

PREFIX = '.'

def top_k(dataset: str, q_idx: int, k: int, self_db: bool = False) -> List[str]:
  save_filename = f'{PREFIX}/data/{dataset}/embeds/score.npy'
  sim_scores = np.load(save_filename)

  if self_db:
    with open(f'{PREFIX}/data/{dataset}/queries.json') as f:
      db_id = json.load(f)[q_idx]['db_id']
  
    tables_df = pd.read_csv(f'{PREFIX}/data/{dataset}/tables.csv')
    table_idxs = tables_df.index[tables_df['db_ids'] == db_id].tolist()
    sim_scores = sim_scores[:, table_idxs]

  if k == -1 or k >= sim_scores.shape[1]:
    top_k_indices = np.argsort(-sim_scores[q_idx])
  else:
    top_k_indices = np.argpartition(-sim_scores[q_idx], k)[:k]
    top_k_indices = top_k_indices[np.argsort(-sim_scores[q_idx][top_k_indices])]

  if self_db:
    top_k_indices = np.array(table_idxs)[top_k_indices]
  tables = pd.read_csv(f'{PREFIX}/data/{dataset}/tables.csv').iloc[top_k_indices]['schema'].tolist()
  db_ids = pd.read_csv(f'{PREFIX}/data/{dataset}/tables.csv').iloc[top_k_indices]['db_ids'].tolist()
  tables = [x.split(',')[0] for x in tables]
  return tables, db_ids

def get_tables(q_idx: int, db_id: Optional[str] = None):
  s = ''

  if db_id is not None:
    with open('./data/bird/schemas.json') as f:
      db = json.load(f)[db_id]
    for table in db:
      s += create_table_statement(db_id, table['name'])
    return s
  
  top_k_tables, top_k_db_ids = top_k('bird', q_idx, 20)
  for (db_id, table) in zip(top_k_db_ids, top_k_tables):
    s += create_table_statement(db_id, table)
  return s

if __name__ == '__main__':
  print(get_tables(1533))