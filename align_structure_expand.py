# Step 2-1: expand the base search objects

from tiger_utils import read_json, write_json, merge
import numpy as np
from tqdm import tqdm
import argparse
import torch

from utils.utils import (
    compatibility_one,
    Table,
    Document,
    get_row_embeds,
    get_chunked_corpus,
    merge_chunk_scores,
    get_segment_idxs,
    EXPAND_KS
)
from utils.bird import load_table_scores
from utils.ottqa import is_doc, retrieve_row
from utils.musique import construct_bm25, get_sent_entity_sim


def expand_base_search_objects(dataset, embedding_model, model, k, num_partitions, partition):
    '''
    Expand the search objects by introducing top-k most compatible neighbors for each base object
    '''

    base_objects_list = read_json(
        f"./keyword_objects/{dataset}/{embedding_model}_{model}/base.json"
    )

    if dataset == "ottqa":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )
        table_embeds, cell_embeds = get_row_embeds(
            dataset, embedding_model, "table"
        ), get_row_embeds(dataset, embedding_model, "cell")
    elif dataset == "musique":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )

        sents_embeds_dict = get_row_embeds(dataset, embedding_model, "sent")
        sents_dict = read_json(f"./data/{dataset}/dev_docs_sentences.json")

        entities_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/entities.npy")
        )
        entities_dict = read_json(f"./data/{dataset}/dev_entities.json")

        segment_idxs = get_segment_idxs(entities_dict)

        bm25 = construct_bm25(entities_dict)
    elif dataset == "wikihop":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )

        sents_embeds_dict = get_row_embeds(dataset, embedding_model, "sent")
        sents_dict = read_json(f"./data/{dataset}/dev_sents.json")

        entities_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/entities.npy")
        )
        entities_dict = read_json(f"./data/{dataset}/dev_entities.json")
        entities = list(entities_dict.values())
        bm25 = construct_bm25(entities)

    scores = np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/score.npy")
    corpus_chunked_objects, object_chunk_idxs = get_chunked_corpus(dataset, "dev")
    objects, _, embed_scores = merge_chunk_scores(
        dataset, corpus_chunked_objects, object_chunk_idxs, scores
    )
    embed_scores = embed_scores.numpy()
    objects_idx = {o: idx for idx, o in enumerate(objects)}

    table_scores = None
    if dataset == "bird":
        table_scores = load_table_scores()

    corpus_objects = {}

    if dataset in ["bird", "ottqa"]:
        corpus_tables = read_json(f"./data/{dataset}/dev_tables.json")
        for t in corpus_tables:
            if dataset == "ottqa":
                corpus_objects[t] = Table(
                    t,
                    t,
                    corpus_tables[t]["header"],
                    rows=corpus_tables[t]["data"],
                    cell_embed=cell_embeds[t].view(-1, 3073).numpy(),
                )
            elif dataset == "bird":
                corpus_objects[t] = Table(
                    t, t, corpus_tables[t]["column_names_original"]
                )

    if dataset != "bird":
        corpus_docs = read_json(f"./data/{dataset}/dev_passages.json")
        for doc_idx, doc_id in enumerate(corpus_docs):
            if dataset in ["ottqa"]:
                corpus_objects[doc_id] = Document(doc_id, doc_id, doc_idx=doc_idx)
            elif dataset == "wikihop":
                corpus_objects[doc_id] = Document(
                    doc_id, doc_id, doc_idx=doc_idx, entity=entities_dict[doc_id]
                )
            elif dataset == "musique":
                doc_name = corpus_docs[doc_id]["title"]
                corpus_objects[doc_id] = Document(
                    doc_id,
                    doc_name,
                    doc_idx=doc_idx,
                    segment_idx=segment_idxs[doc_id],
                    entities=entities_dict[doc_id],
                )

    expanded_objects_list = []

    interval = len(base_objects_list) // num_partitions + 1
    start_idx, end_idx = partition * interval, (partition + 1) * interval

    for q_idx, base_objects in enumerate(tqdm(base_objects_list, total=end_idx-start_idx+1)):
        if not (start_idx <= q_idx < end_idx):
            continue

        remaining_object_names = list(set(corpus_objects.keys()) - set(base_objects))
        remaining_objects = [corpus_objects[o] for o in remaining_object_names]
        remaining_objects_idxs = [objects_idx[o] for o in remaining_object_names]
        rel_scores = embed_scores[q_idx][remaining_objects_idxs]

        expanded_objects = []

        for base_object in base_objects:
            if dataset == "ottqa" and not is_doc(dataset, base_object):
                num_rows_to_keep = 10
                row_idxs = retrieve_row(
                    q_embeds[q_idx], table_embeds[base_object], num_rows_to_keep
                )[1]
                rows = corpus_tables[base_object]["data"]
                cell_embed = (
                    cell_embeds[base_object][row_idxs].reshape(-1, 3073).numpy()
                )
                search_object = Table(
                    base_object,
                    base_object,
                    corpus_tables[base_object]["header"],
                    rows=[rows[idx] for idx in row_idxs],
                    cell_embed=cell_embed,
                )
            elif dataset == "musique":
                num_sents_to_keep = 3
                _sents_embeds, sent_idxs = retrieve_row(
                    q_embeds[q_idx], sents_embeds_dict[base_object], num_sents_to_keep
                )
                _sents = [sents_dict[base_object][sent_idx] for sent_idx in sent_idxs]

                sent_entity_sim = get_sent_entity_sim(
                    _sents, _sents_embeds, entities_embeds, bm25
                )

                search_object = Document(
                    base_object,
                    corpus_docs[base_object]["title"],
                    sent_entity_sim=sent_entity_sim,
                    sents=_sents,
                )
            elif dataset == "wikihop":
                _sents_embeds = sents_embeds_dict[base_object]
                _sents = sents_dict[base_object]

                sent_entity_sim = get_sent_entity_sim(
                    _sents, _sents_embeds, entities_embeds, bm25
                )

                search_object = Document(
                    base_object,
                    base_object,
                    sent_entity_sim=sent_entity_sim,
                    sents=_sents,
                )
            else:
                search_object = corpus_objects[base_object]
            compatibility_scores, connections = compatibility_one(
                dataset, search_object, remaining_objects, table_scores
            )

            scores = rel_scores + compatibility_scores
            top_idxs = np.argsort(-scores)[:k]
            top_objects = [remaining_object_names[top_idx] for top_idx in top_idxs]
            top_connections = [connections[top_idx] for top_idx in top_idxs]
            expanded_objects += top_objects
            expanded_objects.append(base_object)

        expanded_objects_list.append(list(set(expanded_objects)))

    write_json(
        expanded_objects_list,
        f"./results/{dataset}/{embedding_model}_{model}/base_expand_{k}_{partition}.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", type=int)
    parser.add_argument("-d", "--dataset", choices=["bird", "ottqa", "wikihop"])
    parser.add_argument("-embed", "--embedding_model", choices=["uae", "snowflake"])
    parser.add_argument("-lm", "--lm", choices=["llama8", "qwen7"])
    args = parser.parse_args()
    
    if args.dataset == 'bird':
        num_partitions = 10
    elif args.dataset == 'wikihop':
        num_partitions = 20
    elif args.dataset == 'ottqa':
        num_partitions = 40

    # step 1: execute to get the output
    for expand_k in EXPAND_KS[args.dataset]:
        expand_base_search_objects(
            args.dataset, args.embedding_model, args.lm,
            expand_k, num_partitions, args.partition
        )
    
    # step 2: merge the outputs
    # for expand_k in EXPAND_KS[args.dataset]:
    #     merge(num_partitions, f"./results/{args.dataset}/{args.embedding_model}_{args.lm}/base_expand_{expand_k}", "json")
