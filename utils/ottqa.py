from typing import List
import torch.nn.functional as F
from tiger_utils import read_json

COMMON_WORDS = {
  'the', 'in', 'and', 'of', 'is', 'to', 'was', 'as', 'for', 'on', 'by', 'with', 'it', 'at', 'he', 'from', 'an', 'his', 'has', 'its', 'also', 'which', 'that', 'she', 'are', 'were', 'their', 'who', 'they', 'have', 'her', 'or', 'had', 'other', 'but', 'be', 'this', 'between'
}

def get_gold_objects(q, row_as_unit=False):
  target_entities = {q['table_id']} if not row_as_unit else set()

  table_id = q['table_id']

  for ans in q['answer-node']:
    if row_as_unit:
      target_entities.add(f'{table_id}#sep#{ans[1][0]}')

    if ans[-2] is not None and ans[-2].startswith('/wiki'):
      assert ans[-1] == 'passage'
      target_entities.add(ans[-2])
      
  return target_entities

def get_object_original_name(o: str, corpus_objects: List[str]):
  o = o.replace(' ', '_')
  if o in corpus_objects:
    return o
  elif f'/wiki/{o}' in corpus_objects:
    return f'/wiki/{o}'

def is_doc(dataset: str, o: str):
  if dataset in ['musique', 'wikihop']:
    return True
  
  if dataset == 'bird':
    return False

  return o.startswith('/wiki')

def remove_wiki(o: str):
  return o.replace('/wiki/', '')

def retrieve_row(q_embed, rows_embeds, k=None):
  idx = F.cosine_similarity(q_embed.unsqueeze(0), rows_embeds, dim=1).topk(k=1 if k is None else min(k, len(rows_embeds)))[1].tolist()
  return rows_embeds[idx], idx