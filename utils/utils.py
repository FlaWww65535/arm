import torch
from typing import Union
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from tiger_utils import read_json, write_json, concat, read_pickle, write_pickle
import tiktoken

from utils.tokenizer import tokenize

import dotenv
dotenv.load_dotenv(".env")

def create_directory(path):
  parent_dir = os.path.dirname(path)
  if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

class Execute():
  def __init__(self, model_name: str):
    from transformers import AutoTokenizer, set_seed, AutoModelForCausalLM, BitsAndBytesConfig
    from openai import OpenAI

    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype='float16',
    )

    set_seed(0)

    self.model_name = model_name

    if model_name.startswith('gpt'):
      CONFIG = read_json('./config.json')
      self.client = OpenAI(
        api_key=CONFIG['api_key']
      )
    elif model_name.startswith('llama'):
      if model_name == 'llama8':
        model_id = os.environ.get("LLAMA8_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
        print('bfloat16')
        model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map='auto',
          torch_dtype=torch.bfloat16
        )
      else:
        model_id = os.environ.get("LLAMA70B_MODEL_PATH", "meta-llama/Llama-3.1-70B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
          model_id,
          device_map='auto',
          quantization_config=quantization_config,
          local_files_only=True
        )
      tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
      tokenizer.pad_token = tokenizer.eos_token
      self.model = {'model': model, 'tokenizer': tokenizer}
    elif model_name.startswith('qwen'):
      model_id = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
      model = AutoModelForCausalLM.from_pretrained(
          model_id, device_map="auto", torch_dtype=torch.bfloat16, local_files_only=True
      )
      tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
      self.model = {'model': model, 'tokenizer': tokenizer}
  
  def inference(
      self, 
      r_idx:str=None, 
      p:Union[str, list[str]]=None, partial_prefix_allowed_tokens_fn=None, 
      return_logits=False
    ):
    if type(p) is str:
      chat = [{'role': 'user', 'content': p}]
    else:
      chat = p

    if self.model_name.startswith('gpt'):
      resp = self.client.chat.completions.create(
        model=self.model_name,
        temperature=0,
        max_tokens=1024,
        messages=chat
        # stop=[f"\nObservation:"]
      )

      outputs = resp.choices[0].message.content

    if self.model_name.startswith('mistral'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      inputs = tokenizer.apply_chat_template(chat, tokenize=False)
      #print(inputs)
      # default 256 is good except some which do not work, so re-run with 512 for those which do not work
      inputs = tokenizer(inputs, return_tensors='pt').to(model.device)      

      outputs = model.generate(**inputs, max_new_tokens=1024 if self.cot else 512, pad_token_id=tokenizer.eos_token_id)
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      outputs = outputs[0].split('[/INST]')[-1]
      # print(outputs)
      # print('-'*30)

    # prompt need to go into one chunk, can't divide them up
    if self.model_name.startswith('gemma'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      # print(prompt)
      inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
      outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=1024 if self.cot else 256)
      outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      outputs = outputs[0].split('model')[-1]
      # print(outputs)
      # print('-'*30)
    
    if self.model_name.startswith('llama') or self.model_name.startswith('qwen'):
      tokenizer = self.model['tokenizer']
      model = self.model['model']

      prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
      # print(prompt)
      inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

      if self.model_name.startswith('llama'):
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        outputs = model.generate(**inputs,
          max_new_tokens=512, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id,
          prefix_allowed_tokens_fn=partial_prefix_allowed_tokens_fn, stop_strings=['<>'], tokenizer=tokenizer,
          return_dict_in_generate=return_logits, output_logits=return_logits, output_scores=return_logits
        )
      else:
        outputs = model.generate(**inputs,
          max_new_tokens=512,
          prefix_allowed_tokens_fn=partial_prefix_allowed_tokens_fn, stop_strings=['<>'], tokenizer=tokenizer,
          return_dict_in_generate=return_logits, output_logits=return_logits, output_scores=return_logits
        )
      
      input_length = inputs['input_ids'].shape[1]

      if not return_logits:
        # we should not be splitting by 'assistant'
        outputs = outputs[0][input_length:]
        outputs = tokenizer.decode(outputs, skip_special_tokens=True)
      else:
        logits_normalized = model.compute_transition_scores(outputs.sequences, outputs.logits, normalize_logits = True).cpu()
        logits_unnormalized = model.compute_transition_scores(outputs.sequences, outputs.logits, normalize_logits = False).cpu()
        scores_normalized = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits = True).cpu()
        scores_unnormalized = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits = False).cpu()
        generated_tokens = outputs.sequences[:, input_length:][0].cpu()
        generated_sentence = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        assert generated_tokens.shape[0] == logits_normalized.shape[1]

        outputs = {
          'sentence': generated_sentence,
          'tokens': generated_tokens,
          'aux': torch.vstack([logits_normalized, logits_unnormalized, scores_normalized, scores_unnormalized])
        }
    
    return {r_idx: outputs} if r_idx is not None else outputs

def write_jsonl(fn: str, prompts: list[str]):
  # clear the file
  with open(fn, 'w'):
    pass

  for p_idx, prompt in enumerate(prompts):
    if prompt is None:
      continue
    with open(fn, 'a') as f:
      # 128 for ottqa, 1024 for bird
      outputs = {'custom_id': str(p_idx), 'method': 'POST', 'url': '/v1/chat/completions', 'body': {'model': 'gpt-4o-mini', 'messages': prompt, 'temperature': 0, 'max_tokens': 1024}}
      f.write(json.dumps(outputs) + '\n')

def parse_jsonl(fn: str):
  with open(f'{fn}.jsonl') as f:
    preds = [json.loads(line) for line in f.readlines()]
  
  preds_format = {}

  # to handle r2 case where I skip indices
  last_pred_id = -1

  for pred in preds:
    pred_id = pred['custom_id']
    pred_content = pred['response']['body']['choices'][0]['message']['content']
    
    for i in range(last_pred_id, int(pred_id) - 1):
      preds_format[i] = ""
    
    last_pred_id = int(pred_id)
    preds_format[pred_id] = pred_content

  print(len(preds_format))
  write_json(list(preds_format.values()), f'{fn}.json')

class Table:
  def __init__(self, id, name, cols, rows=None, cell_embed=None) -> None:
    self.id, self.name, self.cols, self.rows = id, name, cols, rows
    self.cell_name_embed = cell_embed

  @staticmethod
  def serialize(table, dataset, content=None, rows=None, cols=None):
    uid = table['uid'] if dataset == 'ottqa' else f"{table['db_id']}#sep#{table['table_name_original']}"
    name = table['title'] if dataset == 'ottqa' else f"{table['db_id']}.{table['table_name_original']}"

    result_string = f'Table id: {uid}\nTable name: {name}\n'

    if dataset == 'ottqa':
      desc = f"{table['intro']}\n{table['section_title']}\n{table['section_text']}"
      result_string += f'Table description: {desc}\n'

    if content is None:
      df = pd.DataFrame(rows, columns=cols)
      result_string += f'Table content:\n{df.to_markdown(index=False)}'
    else:
      result_string += f'Table content:\n{content}'

    return result_string

class Document:
  def __init__(self, id, name, tokenized_name=None, doc_idx=None, segment_idx=None, sents=None, tokenized_sents=None, sent_entity_sim=None, entity=None, entities=None, tokenized_entities=None) -> None:
    self.id = id
    self.name, self.tokenized_name, self.sents, self.tokenized_sents = name, tokenized_name, sents, tokenized_sents
    self.doc_idx, self.segment_idx = doc_idx, segment_idx
    self.sent_entity_sim = sent_entity_sim
    self.entity = entity
    self.entities, self.tokenized_entities = entities, tokenized_entities

  @staticmethod
  def serialize(id, name, content):
    return f'Document id: {id}\nDocument name: {name}\nDocument content: {content}'

def overlap_coefficient(s1, s2):
  if type(s1) is str:
    s1 = tokenize(s1)
  if type(s2) is str:
    s2 = tokenize(s2)

  score = len(s1 & s2) / min(len(s1), len(s2))
  return 1.0 if score == 1 else score

def compatibility_document_document(d1: Document, d2: Document):
  scores = d1.sent_entity_sim[:, d2.doc_idx] * np.array([int(d2.entity.lower() in sent.lower()) for sent in d1.sents])
  entities = [[d1.name, d2.name, sent] for sent in d1.sents]

  return scores.max().item(), entities[scores.argmax().item()]

def compatibility_table_document(t: Table, d: Document):  
  d_name = d.name
  # compute max over cell and title
  cells = concat(t.rows)
  assert t.cell_name_embed.shape[0] == len(cells)
  exact_scores = np.array([overlap_coefficient(cell, d_name.replace('/wiki/', '')) for cell in cells])
  
  semantic_scores = t.cell_name_embed[:, d.doc_idx]

  scores = 0.2 * exact_scores + 0.8 * semantic_scores

  return scores.max().item(), cells[int(scores.argmax())]

def get_score(t1, col1, t2, col2, score_dict):
  if f'{t1}-{t2}' in score_dict:
    score = score_dict[f'{t1}-{t2}']
  else:
    score = score_dict[f'{t2}-{t1}']

  if f'{t1}#sep#{col1}-{t2}#sep#{col2}' in score:
    score = score[f'{t1}#sep#{col1}-{t2}#sep#{col2}']
  else:
    score = score[f'{t2}#sep#{col2}-{t1}#sep#{col1}']
  
  return score

def compatibility_table_table(t1: Table, t2: Table, table_scores):
  jaccard, uniqueness = table_scores['jaccard'], table_scores['uniqueness']
  semantic_col_sim, exact_col_sim = table_scores['semantic_col_sim'], table_scores['exact_col_sim']
  
  scores, scores_cols = [], []

  for col1 in t1.cols:
    for col2 in t2.cols:
      u_score = max(uniqueness[f'{t1.name}#sep#{col1}'], uniqueness[f'{t2.name}#sep#{col2}'])
      score = 0.5 * get_score(t1.name, col1, t2.name, col2, jaccard)
      score += 0.5 * (0.5 * get_score(t1.name, col1, t2.name, col2, semantic_col_sim) + 0.5 * get_score(t1.name, col1, t2.name, col2, exact_col_sim))
      score *= u_score
      scores.append(score)
      scores_cols.append([f'{t1.name}#sep#{col1}', f'{t2.name}#sep#{col2}'])
  
  return max(scores), scores_cols[int(np.array(scores).argmax())]

# for draft: given an object, get compatibility with all other objects
def compatibility_one(dataset, search_object: Union[Table, Document], objects, table_scores=None, bidirectional=False):
  scores, connections = [], []

  for o in objects:
    score, connection = 0, None
    if search_object.name == o.name:
      score = 0
    elif isinstance(search_object, Table):
      if isinstance(o, Document):
        score, connection = compatibility_table_document(search_object, o)
      elif isinstance(o, Table) and dataset == 'bird':
        if table_scores:
          score, connection = compatibility_table_table(search_object, o, table_scores)
        else:
          score = 0
    elif isinstance(search_object, Document):
      if isinstance(o, Document) and dataset == 'wikihop':
        score, connection = compatibility_document_document(search_object, o)

        if bidirectional:
          score2, connection2 = compatibility_document_document(o, search_object)
          if score2 > score:
            score, connection = score2, connection2
      elif isinstance(o, Table):
        score, connection = compatibility_table_document(o, search_object)
    
    scores.append(score)
    connections.append(connection)
  
  return np.array(scores), connections

# for ILP: given a list of objects, get inter-compatibility
def compatibility_many(dataset, objects, table_scores=None):
  scores, connections = [], []
  for o in objects:
    score, connection = compatibility_one(dataset, o, objects, table_scores, bidirectional=True)
    scores.append(score)
    connections.append(connection)

  return np.vstack(scores), connections

# chunking
def chunk_id_to_original_id(dataset: str, chunk_id: str):
  if dataset in ['musique', 'wikihop']:
    return chunk_id

  if chunk_id.startswith('/wiki'):
    return chunk_id
  
  return '_'.join(chunk_id.split('_')[:-1])

def get_chunked_table_names(dataset, mode):
  tables_names = [x.replace('.pkl', '') for x in os.listdir(f'./data/{dataset}/{mode}_tables_chunked/') if x.endswith('.pkl')]
  return tables_names

def serialize_object(dataset:str, mode:str, c_id:str, table_row_start_idx=None, table_row_end_idx=None, corpus_docs=None):
  if (dataset == 'ottqa' and c_id.startswith('/wiki/')) or (dataset in ['musique', 'wikihop']):
    docs = corpus_docs if corpus_docs else read_json(f'./data/{dataset}/{mode}_passages.json')

    if dataset in ['ottqa', 'wikihop']:
      o_str = Document.serialize(c_id, c_id, docs[c_id])
    else:
      o_str = Document.serialize(c_id, docs[c_id]['title'], docs[c_id]['content'])
  else:
    tables = read_json(f'./data/{dataset}/{mode}_tables.json')
    if chunk_id_to_original_id(dataset, c_id) in tables:
      table = tables[chunk_id_to_original_id(dataset, c_id)]
    else:
      table = tables[c_id]

    if table_row_start_idx is None:
      chunked_table_markdown = read_pickle(f'./data/{dataset}/{mode}_tables_chunked_markdown/{c_id}.pkl')
      o_str = Table.serialize(table, dataset, content=chunked_table_markdown)
    else:
      if dataset == 'ottqa':
        rows, cols = tables[c_id]['data'], tables[c_id]['header']
      else:
        rows = read_pickle(f'./data/{dataset}/{mode}_rows/{c_id}.pkl')
        cols = tables[c_id]['column_names_original']
      o_str = Table.serialize(table, dataset, rows=rows[table_row_start_idx:table_row_end_idx], cols=cols)

  return o_str

def get_corpus(dataset: str, mode: str):
  if dataset == 'ottqa':
    corpus_tables = read_json(f'./data/ottqa/{mode}_tables.json').keys()
    corpus_docs = read_json(f'./data/ottqa/{mode}_passages.json').keys()
    return list(corpus_tables) + list(corpus_docs)
  
  return list(read_json(f'./data/bird/{mode}_tables.json').keys())

# type is either table or doc
def get_row_embeds(dataset: str, embedding_model:str, type: str):
  if type == 'table':
    rows_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/{embedding_model}/dev/table_rows.npy'))
    objects = read_json('./data/ottqa/dev_tables_rows.json')
  elif type == 'cell':
    rows_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/{embedding_model}/dev/cell_doc_name.npy'))
    objects = read_json('./data/ottqa/dev_tables_cells.json')
  elif type == 'sent_name':
    rows_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/{embedding_model}/dev/sent_doc_name.npy'))
    objects = read_json(f'./data/{dataset}/dev_docs_sentences.json')
  elif type == 'entity':
    rows_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/{embedding_model}/dev/entities.npy'))
    objects = read_json(f'./data/{dataset}/dev_entities.json')
  elif type == 'sent':
    rows_embeds = torch.from_numpy(np.load(f'./data/{dataset}/embeds/{embedding_model}/dev/doc_sents.npy'))
    objects = read_json(f'./data/{dataset}/dev_sents.json')

  rows_embeds_groups = {}
  start_idx = 0
  for t in objects:
    rows = objects[t]
    rows_embeds_groups[t] = rows_embeds[start_idx : start_idx + len(rows)]
    start_idx += len(rows)

  if type == 'cell':
    dev_tables = read_json(f'./data/{dataset}/dev_tables.json')
    for t in dev_tables:
      num_rows, num_cols = len(dev_tables[t]['data']), len(dev_tables[t]['data'][0])
      # 3073 is the number of documents
      rows_embeds_groups[t] = rows_embeds_groups[t].view(num_rows, num_cols, 3073)

  return rows_embeds_groups

def get_segment_idxs(objects_dict):
  segment_idxs = {}
  
  start_idx = 0
  for doc_id in objects_dict:
    segment_idxs[doc_id] = [start_idx, start_idx + len(objects_dict[doc_id])]
    start_idx += len(objects_dict[doc_id])
  
  return segment_idxs

def get_corpus_objects(dataset, embedding_model):
  if dataset == 'ottqa':
    cell_name_sim = get_row_embeds(dataset, embedding_model, 'cell')
  elif dataset == 'musique':
    sents_dict = read_json(f'./data/{dataset}/dev_docs_sentences.json')
    entities_dict = read_json(f'./data/{dataset}/dev_entities.json')
    segment_idxs = get_segment_idxs(entities_dict)
  elif dataset == 'wikihop':
    sents_dict = read_json(f'./data/{dataset}/dev_sents.json')
    entities_dict = read_json(f'./data/{dataset}/dev_entities.json')

  corpus_objects = {}

  if dataset in ['bird', 'ottqa']:
    corpus_tables = read_json(f'./data/{dataset}/dev_tables.json')
    for t in corpus_tables:
      if dataset == 'ottqa':
        corpus_objects[t] = Table(t, t, corpus_tables[t]['header'], rows=corpus_tables[t]['data'], cell_embed=cell_name_sim[t].view(-1, 3073).numpy())
      elif dataset == 'bird':
        corpus_objects[t] = Table(t, t, corpus_tables[t]['column_names_original'])

  if dataset != 'bird':
    corpus_docs = read_json(f'./data/{dataset}/dev_passages.json')
    for doc_idx, doc_id in enumerate(corpus_docs):
      if dataset == 'ottqa':
        corpus_objects[doc_id] = Document(doc_id, name=doc_id, tokenized_name=tokenize(doc_id), doc_idx=doc_idx)
      elif dataset == 'musique':
        doc_name, sents = corpus_docs[doc_id]['title'], sents_dict[doc_id]
        corpus_objects[doc_id] = Document(doc_id, name=doc_name, doc_idx=doc_idx, segment_idx=segment_idxs[doc_id], sents=sents, entities=entities_dict[doc_id])
      elif dataset == 'wikihop':
        corpus_objects[doc_id] = Document(doc_id, name=doc_id, doc_idx=doc_idx, sents=sents_dict[doc_id], entity=entities_dict[doc_id])

  return corpus_objects

def embed(texts, fn, model_name: str, hide_progress=False, model=None):
  if model_name == 'uae':
    return embed_uae(texts, fn, hide_progress, model)
  elif model_name == 'snowflake':
    return embed_snowflake(texts, fn, hide_progress, model)

def embed_uae(texts: list[str], fn: Union[str, None], hide_progress=False, model=None):
  BATCH_SIZE = 200
  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))
  
  from angle_emb import Prompts
  if model is None:
    from angle_emb import AnglE
    model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break

    assert len(_texts) >= 1
    
    _texts = [{'text': text} for text in _texts]
    vec = model.encode(_texts, prompt=Prompts.C)

    embeds.append(vec)
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds

def embed_snowflake(texts: list[str], fn: Union[str, None], hide_progress=False, model=None):
  BATCH_SIZE = 200

  if fn is not None and os.path.isfile(fn):
    return torch.from_numpy(np.load(fn))

  if model is None:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v2.0', trust_remote_code=True).cuda()

  embeds = []
  for i in tqdm(range((len(texts)//BATCH_SIZE) + 1), disable=hide_progress):
    _texts = texts[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

    if len(_texts) == 0:
      break

    assert len(_texts) >= 1

    vec = model.encode(_texts, prompt_name='query', show_progress_bar=False)

    embeds.append(vec)
  
  embeds = np.vstack(embeds)

  if fn is not None:
    np.save(fn, embeds)
  
  embeds = torch.from_numpy(embeds)

  return embeds


def get_chunked_corpus(dataset: str, mode: str):
  object_chunk_idxs = {}

  if dataset in ['ottqa', 'bird']:
    # load ids of chunked tables
    chunked_tables = get_chunked_table_names(dataset, mode)
    chunked_tables = sorted(chunked_tables)

    curr_t_id = None
    for idx, chunked_t_id in enumerate(chunked_tables):
      t_id = chunk_id_to_original_id(dataset, chunked_t_id)

      if t_id != curr_t_id:
        object_chunk_idxs[t_id] = [idx, None]
        if curr_t_id is not None:
          object_chunk_idxs[curr_t_id][1] = idx
        curr_t_id = t_id
    object_chunk_idxs[curr_t_id][1] = len(chunked_tables)

  if dataset in ['ottqa', 'musique', 'wikihop']:
    # load docs
    dev_docs = read_json(f'./data/{dataset}/{mode}_passages.json')
    curr_idx = len(chunked_tables) if dataset == 'ottqa' else 0
    for doc_id in dev_docs:
      object_chunk_idxs[doc_id] = [curr_idx, curr_idx+1]
      curr_idx += 1
    
    if dataset == 'ottqa':
      objects = chunked_tables + list(dev_docs.keys())
    else:
      objects = list(dev_docs.keys())
  else:
    objects = chunked_tables

  return objects, object_chunk_idxs

# we take the maximum for each chunk (start to end), and concat them
def merge_chunk_scores(dataset, corpus_chunked_objects, object_chunk_idxs, scores):
  if len(object_chunk_idxs) == scores.shape[1]:
    return corpus_chunked_objects, None, torch.from_numpy(scores)

  corpus_chunked_objects = np.array(corpus_chunked_objects)

  objects = list(object_chunk_idxs.keys())
  objects_score, chunk_objects = [], []

  num_qs = scores.shape[0]

  for o_id in object_chunk_idxs:
    start_idx, end_idx = object_chunk_idxs[o_id]
    if end_idx - start_idx == 1:
      chunk_objects.append(np.expand_dims(corpus_chunked_objects[np.full(num_qs, start_idx)], axis=1))
      objects_score.append(scores[:, start_idx:end_idx])
    else:
      chunk_objects.append(np.expand_dims(corpus_chunked_objects[start_idx + np.argmax(scores[:, start_idx:end_idx], axis=1)], axis=1))
      objects_score.append(np.expand_dims(np.max(scores[:, start_idx:end_idx], axis=1), axis=1))
  objects_score = torch.from_numpy(np.hstack(objects_score))
  chunk_objects = np.hstack(chunk_objects)

  for _chunk_objects in chunk_objects:
    assert [chunk_id_to_original_id(dataset, x) for x in _chunk_objects] == objects

  return objects, chunk_objects, objects_score


# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def get_num_tokens(texts, tokenizer=None):
  if tokenizer is None:
    enc = tiktoken.encoding_for_model('gpt-4o-mini')

  assert type(texts) in [str, list]
  if type(texts) is str:
    if tokenizer:
      return len(tokenizer(texts)['input_ids'])
    return len(enc.encode(texts))
  
  num_tokens = 0
  for text in texts:
    content = text['content']
    num_tokens += len(tokenizer(content)['input_ids']) if tokenizer else len(enc.encode(content))
  return num_tokens

# 0.150, 0.075, 0.600 / 1M tokens
def get_cost(input_tokens, output_tokens, cached_tokens=None):
  original_cost = 0.150 * input_tokens / 1000000 + 0.600 * output_tokens / 1000000
  if cached_tokens:
    original_cost += 0.075 * cached_tokens / 1000000

  cost = 1000 * original_cost

  print(f'input tokens: {(input_tokens):.2f}', f'output tokens: {(output_tokens):.2f}', f'cost: {cost:.2f}')

  print(' & ' +  ' & '.join([f'{(input_tokens):.1f}', f'{(output_tokens):.1f}', f'{cost:.2f}']))


EXPAND_KS = {
  "bird": [2, 3, 4],
  "ottqa": [3, 4, 5],
  "wikihop": [1, 2, 3]
}