# Step 2-2: for each expanded set of search objects, use mixed-integer programming (MIP) to find the top-k objects

from multiprocessing import Process
import os

from mip import *
import torch
from tqdm import tqdm
import argparse
import numpy as np
import logging
from tiger_utils import read_json, write_json, merge

from utils.utils import (
    compatibility_many,
    get_corpus_objects,
    get_chunked_corpus,
    merge_chunk_scores,
    get_row_embeds,
    get_segment_idxs,
    EXPAND_KS
)
from utils.ottqa import retrieve_row
from utils.bird import load_table_scores
from utils.musique import construct_bm25, get_sent_entity_sim
from utils.logging_utils import ExperimentLogger

SEP = "?sep?"
logger = logging.getLogger(__name__)

def ilp(dataset: str, objects, k: int, rel_scores, table_scores):
    num_objects = len(objects)

    # join score
    cr, connections = compatibility_many(dataset, objects, table_scores)

    model = Model(sense=MAXIMIZE)

    o = [
        model.add_var(var_type=BINARY, name=f"o{SEP}{objects[i].id}")
        for i in range(num_objects)
    ]
    c = [[0 for _ in range(num_objects)] for _ in range(num_objects)]
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                c[i][j] = model.add_var(
                    var_type=BINARY,
                    name=f"c{SEP}{i}{SEP}{j}{SEP}{objects[i].id}{SEP}{objects[j].id}",
                )

    # <= because some cand list does not have k objects
    model += xsum(o[i] for i in range(num_objects)) <= k
    model += xsum(
        c[i][j] for i in range(num_objects) for j in range(num_objects)
    ) <= 2 * (k - 1)

    for i in range(num_objects):
        for j in range(num_objects):
            if i != j:
                model += 2 * c[i][j] <= o[i] + o[j]

    w1, w2 = [2, 1] if dataset == "wikihop" else [3, 1]
    obj = w1 * xsum(rel_scores[i] * o[i] for i in range(num_objects)) + w2 * 0.5 * xsum(
        cr[i][j] * c[i][j] for i in range(num_objects) for j in range(num_objects)
    )

    model.objective = maximize(obj)
    model.verbose = 0
    model.optimize(max_seconds=60)

    # post-process
    pred_objects, pred_joins = [], []
    added_join = []

    for v in model.vars:
        if abs(v.x) > 1e-6:
            if v.name.startswith(f"o{SEP}"):
                _, o_name = v.name.split(SEP)
                pred_objects.append(o_name)
            elif v.name.startswith(f"c{SEP}"):
                _, i1, i2, o1, o2 = v.name.split(SEP)
                i1, i2 = int(i1), int(i2)
                # if i1 < i2:
                #   pred_joins.append([o1, o2, connections[i1][i2], cr[i1][i2]])
                if f"{i1}-{i2}" not in added_join and f"{i2}-{i1}" not in added_join:
                    if cr[i1][i2] > cr[i2][i1]:
                        pred_joins.append([o1, o2, connections[i1][i2], cr[i1][i2]])
                    else:
                        pred_joins.append([o2, o1, connections[i2][i1], cr[i2][i1]])
                    added_join += [f"{i1}-{i2}", f"{i2}-{i1}"]

    return [pred_objects, pred_joins]


def get_rel_score(cand_object, objects_idx, embed_score):
    cand_objects_idxs = [objects_idx[o] for o in cand_object]
    return embed_score[cand_objects_idxs]


def filter_expanded_search_objects(
    embedding_model: str,
    model: str,
    dataset: str,
    expand_k: int,
    num_partitions: int,
    partition: int,
    ilp_k: int = 10,
):
    '''
    Filter the expanded set of search objects from `align_structure_expand.py`
    '''

    draft_fn = f"base_expand_{expand_k}"
    cand_objects = read_json(f"./results/{dataset}/{embedding_model}_{model}/{draft_fn}.json")
    num_qs = len(cand_objects)

    interval = (num_qs // num_partitions) + 1
    start, end = partition * interval, (partition + 1) * interval

    scores = np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/score.npy")
    corpus_chunked_objects, object_chunk_idxs = get_chunked_corpus(dataset, "dev")
    objects, _, embed_scores = merge_chunk_scores(
        dataset, corpus_chunked_objects, object_chunk_idxs, scores
    )
    embed_scores = embed_scores.numpy()
    objects_idx = {o: idx for idx, o in enumerate(objects)}

    corpus_objects_repr = get_corpus_objects(dataset, embedding_model)
    table_scores = load_table_scores() if dataset == "bird" else None
    if dataset == "musique":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )

        sents_embeds_dict = get_row_embeds(dataset, embedding_model, "doc")
        sents_dict = read_json(f"./data/{dataset}/dev_docs_sentences.json")

        entities_embeds_dict = get_row_embeds(dataset, embedding_model, "entity")
        entities_dict = read_json(f"./data/{dataset}/dev_entities.json")

        segment_idxs = get_segment_idxs(entities_dict)

        bm25 = construct_bm25(entities_dict)
    elif dataset == "wikihop":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )

        sents_embeds_dict = get_row_embeds(dataset, embedding_model, "sent")
        sents_dict = read_json(f"./data/{dataset}/dev_sents.json")

        docs = read_json(f"./data/{dataset}/dev_passages.json")
        doc_idxs = {doc_name: doc_idx for doc_idx, doc_name in enumerate(docs)}

        entities_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/entities.npy")
        )
        entities_dict = read_json(f"./data/{dataset}/dev_entities.json")
        entities = list(entities_dict.values())
        bm25 = construct_bm25(entities)

    pred_list = []
    for q_idx in tqdm(range(num_qs), total=end-start+1):
        if not (start <= q_idx < end):
            continue

        if len(cand_objects[q_idx]) == 0:
            pred_list.append([[], []])
            continue

        cand_objects_repr = [corpus_objects_repr[o] for o in cand_objects[q_idx]]

        if dataset == "musique":
            entities_embeds = torch.vstack(
                [
                    entities_embeds_dict[cand_object_repr.id]
                    for cand_object_repr in cand_objects_repr
                ]
            )
            bm25_focus_interval = [
                segment_idxs[cand_object_repr.id]
                for cand_object_repr in cand_objects_repr
            ]
            sub_entities_dict = {
                cand_object_repr.id: entities_dict[cand_object_repr.id]
                for cand_object_repr in cand_objects_repr
            }
            sub_segment_idxs = get_segment_idxs(sub_entities_dict)

            for cand_object_repr in cand_objects_repr:
                doc_id = cand_object_repr.id

                _sents_embeds = sents_embeds_dict[doc_id]
                _sents = sents_dict[doc_id]

                cand_object_repr.sent_entity_sim = get_sent_entity_sim(
                    _sents,
                    _sents_embeds,
                    entities_embeds,
                    bm25,
                    focus_intervals=bm25_focus_interval,
                )
                # overwrite the segment_idxs within this smaller scope
                cand_object_repr.segment_idx = sub_segment_idxs[doc_id]
                # overwrite sentences
                cand_object_repr.sents = _sents
        elif dataset == "wikihop":
            focus_idxs = [
                doc_idxs[cand_object_repr.id] for cand_object_repr in cand_objects_repr
            ]
            sub_entities_embeds = entities_embeds[focus_idxs]

            for cand_object_repr in cand_objects_repr:
                doc_id = cand_object_repr.id

                num_sents_to_keep = 5
                _sents_embeds, sent_idxs = retrieve_row(
                    q_embeds[q_idx], sents_embeds_dict[doc_id], num_sents_to_keep
                )
                _sents = [sents_dict[doc_id][sent_idx] for sent_idx in sent_idxs]

                # _sents_embeds = sents_embeds_dict[doc_id]
                # _sents = sents_dict[doc_id]

                cand_object_repr.sent_entity_sim = get_sent_entity_sim(
                    _sents,
                    _sents_embeds,
                    sub_entities_embeds,
                    bm25,
                    focus_idxs=focus_idxs,
                )
                # overwrite doc_idx within this smaller scope (for efficiency purpose only)
                cand_object_repr.doc_idx = cand_objects[q_idx].index(doc_id)
                cand_object_repr.sents = _sents

        rel_score = get_rel_score(cand_objects[q_idx], objects_idx, embed_scores[q_idx])

        pred_list.append(ilp(dataset, cand_objects_repr, ilp_k, rel_score, table_scores))

    if num_partitions >= 2:
        write_json(
            pred_list,
            f"./results/{dataset}/{embedding_model}_{model}/{draft_fn}_filtered_{partition}.json",
        )
    else:
        write_json(
            pred_list,
            f"./results/{dataset}/{embedding_model}_{model}/{draft_fn}_filtered.json",
        )


def worker(proc_id:int):
    partition_list = list(range(proc_id, num_partitions, args.parallel_num))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(proc_id)
    for partition in partition_list:
        for expand_k in EXPAND_KS[args.dataset]:
            logger.info("expand_k=%d partition=%d/%d", expand_k, partition, num_partitions)
            filter_expanded_search_objects(
                args.embedding_model, args.lm, args.dataset, expand_k, num_partitions, partition
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", type=int)
    parser.add_argument("-parallel_num", "--parallel_num", type=int)
    parser.add_argument("-d", "--dataset", choices=["bird", "ottqa", "wikihop"])
    parser.add_argument("-embed", "--embedding_model", choices=["uae", "snowflake"])
    parser.add_argument("-lm", "--lm", choices=["llama8", "qwen7"])
    args = parser.parse_args()

    logger = ExperimentLogger.configure(
        "align_structure_filter", args.dataset, args.embedding_model, args.lm
    )

    num_partitions = 10

    if args.partition is None:
        #single process 
        if args.parallel_num is not None:
            processes = []
            for proc_id in range(args.parallel_num):
                p = Process(
                    target=worker,
                    args=(proc_id,)
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            for p in processes:
                if p.exitcode != 0:
                    raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")
        else:
            for partition in range(num_partitions):
                for expand_k in EXPAND_KS[args.dataset]:
                    logger.info("expand_k=%d partition=%d/%d", expand_k, partition, num_partitions)
                    filter_expanded_search_objects(
                        args.embedding_model, args.lm, args.dataset, expand_k, num_partitions, partition
                    )
        for expand_k in EXPAND_KS[args.dataset]:
            logger.info("merge expand_k=%d", expand_k)
            merge(
                num_partitions, f'./results/{args.dataset}/{args.embedding_model}_{args.lm}/base_expand_{expand_k}_filtered', 'json'
            )
    else:
        for expand_k in EXPAND_KS[args.dataset]:
            logger.info("expand_k=%d partition=%d/%d", expand_k, args.partition, num_partitions)
            filter_expanded_search_objects(
                args.embedding_model, args.lm, args.dataset, expand_k, num_partitions, args.partition
            )

