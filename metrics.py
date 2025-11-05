import numpy as np
from tqdm import tqdm
from tiger_utils import read_json, read_pickle, write_pickle
import re
import sqlite3
import hashlib
import os
from sql_metadata import Parser

from squad_metrics import compute_exact, compute_f1

def get_p_r_f1(true_positives, false_positives, false_negatives):
  precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
  recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
  f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
  return np.array([precision, recall, f1_score])

class Metrics():
  def __init__(self, top_k = None):
    super().__init__()
    self.tp, self.fp, self.fn, self.acc, self.perfect_recall = 0.0, 0.0, 0.0, [], []
    self.p_r_f1 = np.array([0.0, 0.0, 0.0])
    self.top_k = top_k
  
  def update(self, preds: list[list[str]], target: list[list[str]]):
    cnt, pred_num, tgt_num = 0, 0, 0

    assert len(preds) == len(target)
    for pred, tgt in zip(preds, target):
      pred, tgt = [x.upper() for x in pred], [x.upper() for x in tgt]
      cnt += 1
      pred, tgt = set(pred), set(tgt)

      # treat everything as one block
      self.tp += len(pred & tgt)
      self.fp += len(pred - tgt)
      self.fn += len(tgt - pred)

      # compute independently
      _tp = len(pred & tgt)
      _fp = len(pred - tgt)
      _fn = len(tgt - pred)
      self.p_r_f1 += get_p_r_f1(_tp, _fp, _fn)

      self.acc.append(int(pred == tgt))
      self.perfect_recall.append(int(pred.issuperset(tgt)))

      pred_num += len(pred)
      tgt_num += len(tgt)
    
    print(f'average number of predicted objects: {(pred_num/cnt):.3f}')
    print(f'average number of gold objects: {(tgt_num/cnt):.3f}')
    print(f'#parsable answer: {cnt}')

    # print(f'score (one unit): {self.precision():.1f}, {self.recall():.1f}, {self.f1():.1f}, {self.accuracy():.1f}')
    self.p_r_f1 = np.round(100 * self.p_r_f1 / cnt, 1)
    self.acc = np.round(100*np.array(self.acc).mean(), 1)
    self.perfect_recall = np.round(100*np.array(self.perfect_recall).mean(), 1)
    print(f'score (average): {self.p_r_f1}, {self.acc}, {self.perfect_recall}')

    return np.array([pred_num/cnt] + self.p_r_f1.tolist() + [self.perfect_recall.item()])
  
  def precision(self):
    denominator = self.tp + self.fp
    if denominator == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fp) * 100

  def recall(self):
    denominator = self.tp + self.fn
    if denominator == 0:
      return 0
    else:
      return self.tp / (self.tp + self.fn) * 100
  
  def f1(self):
    denominator = 2 * self.tp + self.fp + self.fn
    if denominator == 0:
      return 0
    else:
      return 2 * self.tp / (2 * self.tp + self.fp + self.fn) * 100

def get_gold_objects(dataset, q):
  if dataset == 'ottqa':
    target_entities = {q['table_id']}

    for ans in q['answer-node']:
      if ans[-2] is not None and ans[-2].startswith('/wiki'):
        assert ans[-1] == 'passage'
        target_entities.add(ans[-2])
        
    return target_entities
  
  if dataset == 'bird':
    gold_ts, db_id = Parser(q['SQL']).tables, q['db_id']
    gold_ts = [f'{db_id}#sep#{gold_t}' for gold_t in gold_ts]
    return gold_ts

  if dataset == 'musique':
    docs = [decomp['document_id'] for decomp in q['question_decomposition']]
    assert len(set(docs)) == len(docs)
    return docs

  if dataset == 'wikihop':
    gold_docs = set([doc[0] for doc in q['supporting_facts']])
    return list(gold_docs)

def get_skip_idxs(dataset):
  SKIP_IDXS = set()

  if dataset == 'ottqa':
    SKIP_IDXS = set()

    for q_idx, q in enumerate(read_json('./data/ottqa/dev.json')):
      if len(get_gold_objects(dataset, q)) == 1:
        SKIP_IDXS.add(q_idx)

  return SKIP_IDXS

def eval_retrieval(dataset, preds, latex_output=False):
  qs = read_json(f'./data/{dataset}/dev.json')
  golds = [get_gold_objects(dataset, q) for q in qs]

  SKIP_IDXS = get_skip_idxs(dataset)
  
  preds_tmp, golds_tmp = [], []
  for q_idx, (pred, gold) in enumerate(zip(preds, golds)):
    # skip questions with only 1 gold object
    if q_idx in SKIP_IDXS:
      continue
    if len(pred) == 0:
      continue
    preds_tmp.append(pred)
    golds_tmp.append(gold)
  
  preds, golds = preds_tmp, golds_tmp
  score = Metrics()
  score = score.update(preds, golds)
  if latex_output:
    score_latex = [str(x) for x in score]
    print(' & '.join(score_latex))
  return score[1:]

def exec_sql(sql: str, cursor):
  hashed_sql = hashlib.sha256(sql.encode('utf-8')).hexdigest()
  sql_fn = f'./sql_outputs/{hashed_sql}.pkl'
  
  if os.path.isfile(sql_fn):
    return read_pickle(sql_fn)

  try:
    cursor.execute(sql)
    rows = cursor.fetchall()
  except:
    rows = None

  write_pickle(rows, sql_fn)
  return rows

def eval_ete_bird(preds):
  qs = read_json(f'./data/bird/dev.json')
  dev_tables = read_json(f'./data/bird/dev_tables.json')
  
  pred_acc, cnt = 0, 0
  empty_idxs = []

  SKIP_IDXS = get_skip_idxs('bird')

  for q_idx, q in enumerate(tqdm(qs)):
    if q_idx in SKIP_IDXS:
      continue

    # skip empty
    if preds[q_idx] == '':
      empty_idxs.append(q_idx)
      continue
    
    # print(q_idx)
    pred = preds[q_idx].replace('\n', ' ')

    cnt += 1

    valid = '<ans>' in pred and '</ans>' in pred
    if valid:
      pred_sql = re.findall(r'<ans>(.*?)</ans>', pred)[0]
    else:
      pred_sql = pred

    gold_db_id, gold_sql = q['db_id'], q['SQL']

    connection = sqlite3.connect(f'./data/bird/dev_database/{gold_db_id}/{gold_db_id}.sqlite')
    cursor = connection.cursor()

    gold_rows = exec_sql(gold_sql, cursor)

    for t in dev_tables:
      replace_strs = [t, f"{dev_tables[t]['db_id']}.{dev_tables[t]['table_name_original']}"]
      for replace_str in replace_strs:
        if replace_str in pred_sql:
          pred_sql = pred_sql.replace(replace_str, dev_tables[t]['table_name_original'])
    
    pred_rows = exec_sql(pred_sql, cursor)

    if pred_rows is not None and set(gold_rows) == set(pred_rows):
      pred_acc += 1
  
  print(len(empty_idxs), empty_idxs)
  print(cnt, f'{(100*pred_acc/cnt):.1f}')

def eval_ete_multihop(dataset: str, preds):
  qs = read_json(f'./data/{dataset}/dev.json')
  assert len(qs) == len(preds)

  exact_list, f1_list = [], []

  SKIP_IDXS = get_skip_idxs(dataset)
  
  for q_idx, q in enumerate(qs):    
    if q_idx in SKIP_IDXS:
      continue

    pred = preds[q_idx]

    # skip empty
    if pred == '':
      continue

    valid = '<ans>' in pred and '</ans>' in pred
    if valid:
      pred_ans = re.findall(r'<ans>(.*?)</ans>', pred)[0]
    else:
      pred_ans = pred

    gold_ans = q['answer-text'] if dataset == 'ottqa' else q['answer']
    
    exact, f1 = compute_exact(gold_ans, pred_ans), compute_f1(gold_ans, pred_ans)
    exact_list.append(exact)
    f1_list.append(f1)
  
  print(f'parsed answer: {len(exact_list)}')
  exact, f1 = 100*np.mean(exact_list), 100*np.mean(f1_list)
  print(f'exact: {exact:.1f}, f1: {f1:.1f}')
  print(f'{exact:.1f} & {f1:.1f}')
  return exact_list, f1_list

# if empty, then skip, instead of using SKIP_IDXS
def eval_ete(dataset, preds):
  if dataset == 'bird':
    eval_ete_bird(preds)
  elif dataset in ['ottqa', 'wikihop']:
    eval_ete_multihop(dataset, preds)