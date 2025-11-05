import torch
import numpy as np
from typing import List
from nltk.stem import PorterStemmer

from utils.ottqa import COMMON_WORDS
from utils.utils import read_json, concat, chunk_id_to_original_id

def completed(token, tokenizer):
  # text = tokenizer.decode(token, skip_special_tokens=True)
  # return text.endswith('|') or text.endswith(' ') or text.endswith('<>')
  return token[-1] in [tokenizer.pad_token_id, tokenizer.eos_token_id]


# do a beam search for one keyword
# beam should be [[token1, token2, ..., token_n]]
# scores should be [beam_score_1, beam_score_2, ..., beam_score_n]
# this function should literally search for the next keyword, using beam search
beam_size = 5
def beam_search(model, tokenizer, text, vocab_ids, remove_keyword_idxs):
  # print(text)
  inputs = tokenizer(text, return_tensors='pt').to(model.device)
  # print(inputs.input_ids.shape, inputs.attention_mask.shape)
  with torch.no_grad():
    outputs = model(**inputs, use_cache=False)
  # print(outputs.logits.shape)
  logits = torch.log_softmax(outputs.logits, dim=-1)[0, -1, :].detach().cpu()

  if len(remove_keyword_idxs) >= 1:
    available_vocabs = torch.ones(vocab_ids.shape[0], dtype=bool)
    available_vocabs[remove_keyword_idxs] = False
    available_tokens = torch.unique(vocab_ids[available_vocabs][:, 0])
  else:
    available_tokens = torch.unique(vocab_ids[:, 0])

  top_k_vals, top_k_idxs = torch.topk(logits[available_tokens], beam_size)
  top_k_idxs = available_tokens[top_k_idxs]

  beams = top_k_idxs.unsqueeze(1).tolist()
  scores = top_k_vals.unsqueeze(1).tolist()
  # print(beams)
  # print(tokenizer.batch_decode(beams))
  # print(scores)

  # stop when all beams ends with the separator

  # pad all beams with EOS token
  while not all(completed(beam, tokenizer) for beam in beams):
    completed_beams, completed_scores = [], []
    uncompleted_beams, uncompleted_scores = [], []

    for (beam, score) in zip(beams, scores):
      # in qwen, pad token and eos token are two separate things
      # however, in llama, we set pad_token = eos_token, so effectively we are stopping at pad_token/ eos_token
      if completed(beam, tokenizer):
        completed_beams.append(beam)
        completed_scores.append(score)
      else:
        uncompleted_beams.append(beam)
        uncompleted_scores.append(score)
    
    uncompleted_beams = torch.tensor(uncompleted_beams).to(model.device)
    uncompleted_scores = torch.tensor(uncompleted_scores)

    input_ids_new = torch.hstack([inputs.input_ids.repeat(len(uncompleted_beams), 1), uncompleted_beams])
    attention_masks_new = torch.ones(input_ids_new.shape).to(model.device)

    # the sequence with the beam appended
    with torch.no_grad():
      logits = model(input_ids=input_ids_new, attention_mask=attention_masks_new, use_cache=False).logits.detach().cpu()
    logits = torch.log_softmax(logits, dim=-1)[:, -1, :]
    assert len(logits) == len(uncompleted_beams)

    uncompleted_beams = uncompleted_beams.cpu()

    tmp_beams, tmp_scores = [], []

    # first get all beams that need to compute the logits
    for beam_idx, beam in enumerate(uncompleted_beams):
      # print(vocab_ids[:, :len(beam)].shape, beam.unsqueeze(0).shape)
      # old way: (vocab_ids[:, :len(beam)] == beam.unsqueeze(0)).all(-1)
      # get all vocabs that share the same prefix as beam
      # cannot use .sum(dim=1) == 0 because this doesn't force every entry to be 0 (e.g., -1, 1 sum is also 0)
      # adding absolute value solves the problem
      valid_vocab_bool = (vocab_ids[:, :len(beam)] - beam).abs().sum(dim=1) == 0
      # remove existing keywords
      if len(remove_keyword_idxs) >= 1:
        valid_vocab_bool[remove_keyword_idxs] = False
      available_tokens = vocab_ids[valid_vocab_bool][:, len(beam)]
      

      available_tokens = torch.unique(available_tokens)
      # print(f'available_tokens: {available_tokens}')
      # print(f'#available tokens: {len(available_tokens)}')
      # get the score and then re-rank
      logits_tokens = logits[beam_idx][available_tokens]
      k = min(beam_size, len(available_tokens))
      top_k_vals, top_k_idxs = torch.topk(logits_tokens, k=k)
      
      tmp_beams.append(torch.hstack([beam.repeat(k, 1), available_tokens[top_k_idxs].unsqueeze(1)]))
      tmp_scores.append(torch.hstack([uncompleted_scores[beam_idx].repeat(k, 1), top_k_vals.unsqueeze(1)]))
    
    # the EOS token drags the probability down by a lot, so skip it when computing scores
    tmp_beams = torch.vstack(tmp_beams).tolist() + completed_beams
    tmp_scores = torch.vstack(tmp_scores).tolist() + completed_scores


    # tmp_avg_scores = [sum(s[:-1])/len(s[:-1]) if tmp_beams[i][-1] == eos_token else sum(s)/len(s) for i, s in enumerate(tmp_scores)]
    tmp_avg_scores = [sum(s[:-1])/len(s[:-1]) if completed(tmp_beams[i], tokenizer) else sum(s)/len(s) for i, s in enumerate(tmp_scores)]
    _, top_k_idxs = torch.topk(torch.tensor(tmp_avg_scores), k=beam_size)
    beams = [tmp_beams[idx] for idx in top_k_idxs]
    scores = [tmp_scores[idx] for idx in top_k_idxs]

    # print(tokenizer.batch_decode(beams))
    # print(scores)
    # print('\n\n')
  
  # [:-1] to remove EOS token
  avg_scores = np.array([sum(s[:-1])/len(s[:-1]) for s in scores])
  # print(beams[np.argmax(avg_scores)])
  keyword = tokenizer.decode(beams[np.argmax(avg_scores)], skip_special_tokens=True)
  # print(keyword)
  return keyword

def get_token_id(tokenizer, s: List[str], model_name):
  if model_name.startswith('llama'):
    # [1:] to skip the llama model <begin_of_text> token
    return [x[1:] for x in tokenizer(s)['input_ids']]

  return [x for x in tokenizer(s)['input_ids']]

def match_next_token(sent: List[int], next_tokens: List[List[int]], num_tokens: int) -> List[int]:
  cands_tokens = []
  
  # # optimization for 'Candidate objects: ' long string when there is only one option
  # if len(next_tokens) == 1:
  #   next_token = next_tokens[0]
  #   if num_tokens == len(next_token):
  #     return None
  #   return [next_token[num_tokens]]

  for next_token in next_tokens:
    # this word is completely outputted
    if sent[-num_tokens:] == next_token:
      return None
    if sent[-num_tokens:] == next_token[:num_tokens]:
      cands_tokens.append(next_token[num_tokens])

  assert len(cands_tokens) >= 1
  return cands_tokens

def get_vocab_for_q(q_idx: int, vocab_dict, embeds_ranking, tokenizer, model_name: str):
  # print(len(embeds_ranking[q_idx]))
  vocab = set()
  for e in vocab_dict:
    assert len(embeds_ranking[q_idx]) == 50
    if e in embeds_ranking[q_idx][:50]:
      vocab.update(vocab_dict[e])
  vocab = vocab - {'', ' '} - COMMON_WORDS

  vocab_list = list(vocab)
  vocab_list = [f' {v}' for v in vocab_list]
  vocab_ids_orig = tokenizer(vocab_list, return_tensors='pt', padding=True).input_ids
  
  if model_name.startswith('llama'):
    vocab_ids = torch.hstack([vocab_ids_orig[:, 1:], torch.full((len(vocab_list), 1), tokenizer.eos_token_id)])
  else:
    vocab_ids = torch.hstack([vocab_ids_orig, torch.full((len(vocab_list), 1), tokenizer.eos_token_id)])
  
  stemmer = PorterStemmer()
  vocab_list_stemmed = [stemmer.stem(v) for v in vocab_list]
  # print(f'#vocab: {len(vocab_list)}')

  return vocab_ids, vocab_list, vocab_list_stemmed


# https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate.prefix_allowed_tokens_fn
# https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L1283
# all database, column, and tables should still use add_ticks to signal stop, to prevent one candidate token being a prefix of another candidate
# TODO: add state tracking (i.e., if database is chosen to be xxx, then tables afterwards can only be chosen from that database)
class ConstraintDecoder():
  # def __init__(self, tokenizer, model, interval, start, end, entity):
  def __init__(self, tokenizer, model, dataset, model_name, embedding_model):
    self.model, self.tokenizer = model, tokenizer
    self.model_name, self.embedding_model = model_name, embedding_model
    self.next_tokens, self.num_tokens = [], 0

    self.vocab_dict = read_json(f'./data/{dataset}/vocab_dev.json')
    self.embeds_ranking = read_json(f'./data/{dataset}/embeds/{embedding_model}_pred_dev.json')
    self.embeds_ranking = [[chunk_id_to_original_id(dataset, x) for x in pred[0]] for pred in self.embeds_ranking]

    self.qs = read_json(f'./data/{dataset}/dev.json')

    self.vocab_ids, self.q_idx = None, None
    self.keywords_idxs, self.corpus_words, self.cand_objects = {}, None, None

  def reset(self, q_idx):
    self.next_tokens, self.num_tokens = [], 0
    self.vocab_ids, self.vocab_list, self.vocab_list_stemmed = get_vocab_for_q(q_idx, self.vocab_dict, self.embeds_ranking, self.tokenizer, self.model_name)
    self.q_idx = q_idx
    self.keywords_idxs, self.corpus_words, self.cand_objects = {}, None, None

  def get_next_token(self, batch_id: int, sent: List[int]):
    # check if sent ends in tokens that match any next_token
    # decide when to terminate and reset self.next_tokens = [] and self.num_tokens
    if len(self.next_tokens) >= 1:
      match_tokens = match_next_token(sent, self.next_tokens, self.num_tokens)
      if match_tokens is None:
        self.next_tokens = []
        self.num_tokens = 0
      else:
        self.num_tokens += 1
        return match_tokens

    sent_text: str = self.tokenizer.decode(sent)
    pred_sent_text = sent_text.split('\n')[-1]
    
    next_words = None

    # determine if model is in the keyword generation mode
    in_keyword_phase = 'Candidate objects:' not in pred_sent_text and pred_sent_text.count('(') != pred_sent_text.count(')')

    if in_keyword_phase:
      # print(pred_sent_text)
      # keyword_idx = pred_sent_text.count('(') - 1
      similar_words_str = pred_sent_text.split(' #sep# ')[1]
      assert 'Similar words' in similar_words_str
      similar_words_str = similar_words_str.replace('Similar words: ', '')
      keyword_idx = similar_words_str.count(' | ')

      similar_words = similar_words_str.split(' | ')[keyword_idx]
      keyword = similar_words[:similar_words.rindex('(')].strip()

      keyword_len = len(keyword.split(' '))
      
      enforce_decomp_len = False
      
      if enforce_decomp_len:
        if self.corpus_words is None:
          self.corpus_words = [[] for _ in range(len(keywords))]
      else:
        if self.corpus_words is None:
          self.corpus_words = [[]]
        else:
          self.corpus_words.append([])

      next_words = []

      cand_word = beam_search(self.model, self.tokenizer, sent_text, self.vocab_ids, list(set(concat(self.keywords_idxs.values()))))

      stemmer = PorterStemmer()
      cand_word_stemmed = stemmer.stem(cand_word)
      
      # remove keywords with the same stem (this is superset of remove the selected keywords only)
      if cand_word_stemmed not in self.keywords_idxs:
        self.keywords_idxs[cand_word_stemmed] = [i for i, v in enumerate(self.vocab_list_stemmed) if v == cand_word_stemmed]
      # print(self.keywords_idxs)

      # standalone tokens: )   ),   ,   ' ('
      # 'Ġ(', 'Ġhome', ',', 'Ġstadium', ')', 'Ġ|', 'Ġbr', 'ay', 'Ġwander', 'ers', 'Ġ(', 'Ġbr', 'ay', ',', 'Ġwander', 'ers', ')', 'Ġ#', 'sep'
      # basically, when , is followed by a space by a word, ',' is standalone
      
      # if 4 corpus_words, stop (For bird and ottqa, was 3)
      if len(self.corpus_words[keyword_idx]) == 5:
        next_words += [f'{cand_word}) <>', f'{cand_word}) |']
      # if 0 - keywords_len-1, continue
      elif len(self.corpus_words[keyword_idx]) <= keyword_len - 2:
        next_words += [f'{cand_word},']
      else:
        next_words += [f'{cand_word},', f'{cand_word}) <>', f'{cand_word}) |']
      
      self.corpus_words[keyword_idx].append(cand_word)

    if next_words is not None:
      self.next_tokens = get_token_id(self.tokenizer, next_words, self.model_name)
      self.num_tokens += 1
      return [x[0] for x in self.next_tokens]

    return list(range(self.tokenizer.vocab_size))

# def prefix_allowed_tokens_fn(_batch_id, sent, constraint_decoders, tokenizer, interval, start, end):
def prefix_allowed_tokens_fn(_batch_id, sent, constraint_decoder):
  # if _batch_id not in constraint_decoders:
  #   constraint_decoders[_batch_id] = ConstraintDecoder(tokenizer, interval, start, end)
  # allowed_tokens = constraint_decoders[_batch_id].get_next_token(_batch_id, sent.tolist())
  allowed_tokens = constraint_decoder.get_next_token(_batch_id, sent.tolist())
  return allowed_tokens