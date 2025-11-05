from tiger_utils import concat, cosine_sim
from bm25s import BM25
import numpy as np

from utils.tokenizer import tokenize

def get_gold_objects(q):
  docs = [decomp['document_id'] for decomp in q['question_decomposition']]
  assert len(set(docs)) == len(docs)
  return docs

def construct_bm25(object_dict):
  if type(object_dict) is not list:
    objects = concat(list(object_dict.values()))
  else:
    objects = object_dict

  tokenized_objects = [list(tokenize(o)) for o in objects]
  bm25 = BM25(b=0)
  bm25.index(tokenized_objects, show_progress=True)
  return bm25

def get_bm25_score(bm25, tokenized_sent):
  score = bm25.get_scores(list(tokenized_sent))
  if score.max() > 0:
    score /= score.max()
  score = np.expand_dims(score, axis=0)
  return score

def get_sent_entity_sim(sents, sents_embeds, entities_embeds, bm25, focus_intervals=None, focus_idxs=None):
  sent_entity_semantic_sim = cosine_sim(sents_embeds, entities_embeds).numpy()
  tokenized_sents = [tokenize(sent) for sent in sents]
  sent_entity_exact_sim = np.vstack([get_bm25_score(bm25, tokenized_sent) for tokenized_sent in tokenized_sents])
  
  if focus_intervals:
    sent_entity_exact_sim = [sent_entity_exact_sim[:, start:end] for start, end in focus_intervals]
    sent_entity_exact_sim = np.hstack(sent_entity_exact_sim)

  if focus_idxs:
    sent_entity_exact_sim = sent_entity_exact_sim[:, focus_idxs]

  assert sent_entity_semantic_sim.shape == sent_entity_exact_sim.shape
  sent_entity_sim = 0.4 * sent_entity_semantic_sim + 0.6 * sent_entity_exact_sim

  return sent_entity_sim