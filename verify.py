# Step 3-1: self-verification

import argparse
from transformers import set_seed
import re
import logging
from tqdm import tqdm
import torch
import numpy as np
from tiger_utils import read_json, write_json, concat, read_pickle, merge
from tiger_utils.model import system, user, assistant

from utils.utils import (
    chunk_id_to_original_id,
    merge_chunk_scores,
    get_chunked_corpus,
    serialize_object,
    get_row_embeds,
    Table,
    Document,
    EXPAND_KS
)
from utils.ottqa import retrieve_row, is_doc
from metrics import eval_retrieval
from utils.execute import run_prompts
from utils.logging_utils import ExperimentLogger

logger = logging.getLogger(__name__)

def get_join(dataset, j):
    if dataset == "ottqa":
        if is_doc(dataset, j[0]):
            d, t = j[0], j[1]
        else:
            d, t = j[1], j[0]

        return f"{t} includes {j[2]} which connects with {d}"

    if dataset == "wikihop":
        d1, d2, e = j[2]
        return f'{d1} includes sentence "{e}", which connects with {d2}'

    col1, col2 = j[2]
    db1, t1, col1 = col1.split("#sep#")
    db2, t2, col2 = col2.split("#sep#")
    return (
        f"column {col1} in {db1}#sep#{t1} connects with column {col2} in {db2}#sep#{t2}"
    )


def get_user_prompt(dataset, q, keywords, relevant_words, chunks_ser, join_info):
    keywords = keywords.replace("Keywords: ", "")
    relevant_words = relevant_words.replace("Similar words: ", "")
    join_info = [get_join(dataset, j) for j in join_info]
    user_str = "\n\n".join(
        [
            f"User question: {q}",
            f"Keywords: {keywords}",
            f"Similar words: {relevant_words}",
            "\n\n".join(chunks_ser),
            "\n".join(join_info),
        ]
    )
    return user(user_str)


def vote_filtered_search_objects(
    dataset: str, model: str, embedding_model: str, num_partitions, partition, ilp_fn
):
    '''
    Use LLM to vote for objects from `align_structure_filter.py` in the form of logits
    '''

    print(dataset, model, embedding_model, ilp_fn, num_partitions, partition)
    preds = read_json(f"./results/{dataset}/{embedding_model}_{model}/{ilp_fn}.json")
    qs = read_json(f"./data/{dataset}/dev.json")

    prompts = []

    interval = (len(qs) // num_partitions) + 1
    start, end = partition * interval, (partition + 1) * interval
    q_idxs = list(range(len(qs)))

    examples = read_json(f"./data/{dataset}/examples.json")
    assert len(examples) == 3

    TYPE = "objects"

    if model == "llama8":
        base_prompt = [
            system(
                f"You are given a user question, your task is to decompose the user question into contiguous, non-overlapping substrings that can cover different information mentioned in the user question. For each substring, generate words that are the most relevant to the substring. Based on the generated relevant words, generate a list of relevant {TYPE}, including their names, content, and connections between these {TYPE}. From these candidate {TYPE}, you should identify the minimum number of {TYPE} that can be used to answer the user question based on the relevance between the object name, object content and user question as well as the relevance of the object connections. You should end your response with <>."
            )
        ]
    elif model == "qwen7":
        base_prompt = [
            system(
                f"You are given a user question, your task is to decompose the user question into contiguous, non-overlapping substrings that can cover different information mentioned in the user question. For each substring, generate words that are the most relevant to the substring. Based on the generated relevant words, generate a list of relevant {TYPE}, including their names, content, and connections between these {TYPE}. From these candidate {TYPE}, you should identify the minimum number of {TYPE} that can be used to answer the user question based on the relevance between the object name, object content and user question as well as the relevance of the object connections. Please strictly follow the response format of the examples below. You should end your response with <>."
            )
        ]

    if dataset in ["bird", "ottqa"]:
        train_tables = read_json(f"./data/{dataset}/train_tables.json")

    if dataset != "bird":
        train_docs = read_json(f"./data/{dataset}/train_passages.json")

    for example in examples:
        chunks_ser = []
        for chunk in example["candidate chunks"]:
            c_id = chunk[0]
            if is_doc(dataset, c_id):
                chunks_ser.append(
                    serialize_object(dataset, "train", c_id, corpus_docs=train_docs)
                )
            else:
                if len(chunk) == 1:
                    row_start_idx, row_end_idx = 0, 2
                else:
                    row_start_idx, row_end_idx = chunk[1], chunk[2]

                if dataset == "ottqa":
                    rows, cols = (
                        train_tables[c_id]["data"],
                        train_tables[c_id]["header"],
                    )
                else:
                    rows = read_pickle(f"./data/bird/train_rows/{c_id}.pkl")
                    cols = train_tables[c_id]["column_names_original"]

                chunks_ser.append(
                    Table.serialize(
                        train_tables[c_id],
                        dataset,
                        rows=rows[row_start_idx:row_end_idx],
                        cols=cols,
                    )
                )
        base_prompt.append(
            get_user_prompt(
                dataset,
                example["question"],
                example["keywords"],
                example["similar words"],
                chunks_ser,
                example["candidate joins"],
            )
        )
        base_prompt.append(assistant(" | ".join(example["relevance"]) + " <>"))

    prompts = []

    keywords_list = read_json(f"./results/{dataset}/{embedding_model}_{model}/ia.json")

    if dataset == "ottqa":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )
        table_rows_embeds, sents_embeds = get_row_embeds(
            dataset, embedding_model, "table"
        ), get_row_embeds(dataset, embedding_model, "sent")
        dev_tables, sents_dict = read_json(
            f"./data/{dataset}/dev_tables.json"
        ), read_json(f"./data/{dataset}/dev_sents.json")
    elif dataset == "musique":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )
        doc_sents_embeds = get_row_embeds(dataset, "sent")
        sents_dict = read_json(f"./data/{dataset}/dev_docs_sentences.json")
        docs_dict = read_json(f"./data/{dataset}/dev_passages.json")
    elif dataset == "wikihop":
        q_embeds = torch.from_numpy(
            np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/q_embeds.npy")
        )
        sents_embeds = get_row_embeds(dataset, embedding_model, "sent")
        sents_dict = read_json(f"./data/{dataset}/dev_sents.json")
        docs_dict = read_json(f"./data/{dataset}/dev_passages.json")
    else:
        dev_tables = read_json(f"./data/{dataset}/dev_tables.json")
        # this is if we are using the chunk itself
        corpus_chunked_objects, object_chunk_idxs = get_chunked_corpus(dataset, "dev")
        scores = np.load(f"./data/{dataset}/embeds/{embedding_model}/dev/score.npy")
        objects, chunked_objects, _ = merge_chunk_scores(
            dataset, corpus_chunked_objects, object_chunk_idxs, scores
        )
        objects_idx = {o: idx for idx, o in enumerate(objects)}

    for q_idx, q in enumerate(tqdm(qs)):
        if not (start <= q_idx < end):
            prompts.append(None)
            continue

        pred_objects, pred_join = preds[q_idx]

        if len(pred_objects) == 0:
            prompts.append(None)
        else:
            # this is if we are using the chunk itself

            pred_objects_ser = []

            num_rows = 3 if dataset == "musique" else 5
            if dataset == "wikihop":
                num_rows = 5

            if dataset == "bird":
                pred_chunked_objects = [
                    chunked_objects[q_idx][objects_idx[o]] for o in pred_objects
                ]

                for c_id in pred_chunked_objects:
                    table = dev_tables[chunk_id_to_original_id(dataset, c_id)]
                    rows = read_pickle(
                        f"./data/{dataset}/dev_tables_chunked/{c_id}.pkl"
                    )[:num_rows]
                    chunk_str = Table.serialize(
                        table, dataset, rows=rows, cols=table["column_names_original"]
                    )
                    pred_objects_ser.append(chunk_str)
            else:
                for o in pred_objects:
                    if is_doc(dataset, o):
                        doc_sents = sents_dict[o]
                        _, sents_idxs = retrieve_row(
                            q_embeds[q_idx], sents_embeds[o], k=num_rows
                        )

                        if dataset in ["musique", "wikihop"]:
                            doc_sents = " ".join([doc_sents[idx] for idx in sents_idxs])
                        else:
                            doc_sents = ". ".join(
                                [doc_sents[idx] for idx in sents_idxs]
                            )

                        pred_objects_ser.append(
                            Document.serialize(o, o, doc_sents)
                            if dataset in ["ottqa", "wikihop"]
                            else Document.serialize(o, docs_dict[o]["title"], doc_sents)
                        )
                    else:
                        table_rows = dev_tables[o]["data"]
                        _, rows_idxs = retrieve_row(
                            q_embeds[q_idx], table_rows_embeds[o], k=num_rows
                        )
                        table_rows = [table_rows[idx] for idx in rows_idxs]
                        pred_objects_ser.append(
                            Table.serialize(
                                dev_tables[o],
                                dataset,
                                rows=table_rows,
                                cols=dev_tables[o]["header"],
                            )
                        )

            keywords, relevant_words = (
                keywords_list[q_idx].replace(" <>", "").split(" #sep# ")
            )
            prompts.append(
                base_prompt
                + [
                    get_user_prompt(
                        dataset,
                        q["question"],
                        keywords,
                        relevant_words,
                        pred_objects_ser,
                        pred_join,
                    )
                ]
            )

    run_prompts(
        num_partitions,
        partition,
        prompts,
        f"./results/{dataset}/{embedding_model}_{model}/verify",
        model,
        q_idxs=q_idxs,
        stage="rerank",
        dataset=dataset,
        ilp_fn=ilp_fn,
        return_logits=True,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", type=int)
    parser.add_argument("-k", "--expand_k", type=int)
    parser.add_argument("-d", "--dataset", choices=["bird", "ottqa", "wikihop"])
    parser.add_argument("-embed", "--embedding_model", choices=["uae", "snowflake"])
    parser.add_argument("-lm", "--lm", choices=["llama8", "qwen7"])
    args = parser.parse_args()

    logger = ExperimentLogger.configure("verify", args.dataset, args.embedding_model, args.lm)

    num_partitions = 8
    if args.partition is None:
        for partition in range(num_partitions):
            logger.info("expand_k=%s partition=%d/%d", args.expand_k, partition, num_partitions)
            vote_filtered_search_objects(args.dataset, args.lm, args.embedding_model, num_partitions, partition, f'base_expand_{args.expand_k}_filtered')

        for expand_k in EXPAND_KS[args.dataset]:
            logger.info("merge expand_k=%d", expand_k)
            merge(num_partitions, f'./results/{args.dataset}/{args.embedding_model}_{args.lm}/verify_base_expand_{expand_k}_filtered', 'json')
            merge(num_partitions, f'./results/{args.dataset}/{args.embedding_model}_{args.lm}/verify_aux_base_expand_{expand_k}_filtered', 'pkl')
    else:
        logger.info("expand_k=%s partition=%d/%d", args.expand_k, args.partition, num_partitions)
        vote_filtered_search_objects(args.dataset, args.lm, args.embedding_model, num_partitions, args.partition, f'base_expand_{args.expand_k}_filtered')