from typing import List
import re

from utils.utils import read_json
from utils.ottqa import is_doc

def get_token_id(tokenizer, s: List[str], model_name):
  if model_name.startswith('llama'):
    # [1:] to skip the llama model <begin_of_text> token
    return [x[1:] for x in tokenizer(s)['input_ids']]

  return [x for x in tokenizer(s)['input_ids']]

def match_next_token(sent: List[int], next_tokens: List[List[int]], num_tokens: int) -> List[int]:
  cands_tokens = []

  for next_token in next_tokens:
    # this word is completely outputted
    if sent[-num_tokens:] == next_token:
      return None
    if sent[-num_tokens:] == next_token[:num_tokens]:
      cands_tokens.append(next_token[num_tokens])

  assert len(cands_tokens) >= 1
  return cands_tokens

def get_joinable_objects(objects: List[str], join):
  join = [j for j in join if j[-1] > 0]
  joinable_objects = set()
  for o in objects:
    for j in join:
      if o in j:
        joinable_objects.update(j[:2])
  
  joinable_objects = joinable_objects - set(objects)

  return joinable_objects

def get_joinable_objects_bird(objects: List[str], join):
  join = [x[2].split('-') for x in join]
  joinable_objects = set()
  for o in objects:
    for j in join:
      if o in j:
        joinable_objects.update(j)
  joinable_objects -= set(objects)
  return [x for x in joinable_objects if '#sep#' in x]

def filter_objects(pred_sent_text: str, dataset, objects):
  if pred_sent_text.endswith('<table>'):
    filtered_objects = [f' {o} </table>' for o in objects if not is_doc(dataset, o)]
  else:
    filtered_objects = [f' {o} </document>' for o in objects if is_doc(dataset, o)]
  return filtered_objects


class ConstraintDecoderRerank():
  def __init__(self, tokenizer, dataset: str, ilp_fn: str, model_name: str, embedding_model: str):
    self.tokenizer, self.dataset = tokenizer, dataset
    self.model_name = model_name
    self.next_tokens, self.num_tokens = [], 0

    self.qs = read_json(f'./data/{dataset}/dev.json')
    self.cand_objects = [x[0] for x in read_json(f'./results/{dataset}/{embedding_model}_{model_name}/{ilp_fn}.json')]
    self.cand_joins = [x[1] for x in read_json(f'./results/{dataset}/{embedding_model}_{model_name}/{ilp_fn}.json')]

    self.vocab_ids, self.vocab_list, self.q_idx = None, None, None

  def reset(self, q_idx: int):
    self.next_tokens, self.num_tokens = [], 0
    self.q_idx = q_idx
  
  def get_next_token(self, batch_id: int, sent: List[int]):
    if len(self.next_tokens) >= 1:
      match_tokens = match_next_token(sent, self.next_tokens, self.num_tokens)
      if match_tokens is None:
        self.next_tokens = []
        self.num_tokens = 0
      else:
        self.num_tokens += 1
        return match_tokens

    sent_text: str = self.tokenizer.decode(sent)
    pred_sent_text = sent_text.split('\n\n')[-1]

    next_words = []

    selected_objects = []

    if pred_sent_text.endswith('<table>') or pred_sent_text.endswith('<document>'):
      selected_objects = []
      if '</table>' in pred_sent_text:
        selected_objects += re.findall(r'<table>(.*?)</table>', pred_sent_text)
      
      if '</document>' in pred_sent_text:
        selected_objects += re.findall(r'<document>(.*?)</document>', pred_sent_text)
      
      selected_objects = [x.strip() for x in selected_objects]

      # if no objects have been selected, then let model choose anyone
      if len(selected_objects) == 0:
        next_words += filter_objects(pred_sent_text, self.dataset, self.cand_objects[self.q_idx])
      else:
        # if an object has been selected, then select from the set of objects it joins with

        joinable_objects = set(self.cand_objects[self.q_idx]) - set(selected_objects)
        
        joinable_objects = filter_objects(pred_sent_text, self.dataset, joinable_objects)

        if len(joinable_objects) == 0:
          next_words = ['<>']
        else:
          next_words += joinable_objects
    
    if len(next_words) >= 1:
      self.next_tokens = get_token_id(self.tokenizer, next_words, self.model_name)
      self.num_tokens += 1
      return [x[0] for x in self.next_tokens]

    return list(self.tokenizer.get_vocab().values())