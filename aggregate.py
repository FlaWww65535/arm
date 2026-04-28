# Step 3-2: aggregation

from tiger_utils import read_json, read_pickle, write_json
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import logging

from utils.utils import EXPAND_KS
from utils.logging_utils import ExperimentLogger

logger = logging.getLogger(__name__)

def get_object_segments(tokenizer, token_ids):
    segments = []

    start_idx, seen_close_tag = None, False

    # object name (including tags)
    for idx, token_id in enumerate(token_ids):
        token = tokenizer.decode(token_id)
        if "<" in token and start_idx is None:
            start_idx = idx

        if "</" in token:
            seen_close_tag = True

        if ">" in token and seen_close_tag:
            end_idx = idx
            segments.append([start_idx, end_idx + 1])
            start_idx, seen_close_tag = None, False

    return segments


def get_object_score(token_ids, scores, segments, tokenizer):
    remove_strs = [
        "<table>",
        "</table>",
        "<document>",
        "</document>",
        "<wiki>",
        "</wiki>",
    ]
    scores_dict = {}
    for segment in segments:
        start, end = segment
        object = tokenizer.decode(token_ids[start:end]).strip()

        for s in remove_strs:
            object = object.replace(s, "")
        object = object.strip()

        if object == "" or "<" in object:
            continue

        object_score = scores[start:end].mean()
        scores_dict[object] = object_score

    return scores_dict


def merge_scores(dataset, lm, embedding_model, scores_dicts):
    scores_dict_merged = {}
    for scores_dict in scores_dicts:
        for object in scores_dict:
            if object not in scores_dict_merged:
                scores_dict_merged[object] = [scores_dict[object]]
            else:
                scores_dict_merged[object].append(scores_dict[object])

    objects = list(scores_dict_merged.keys())
    logits_scores, vote_scores = [], []

    for o in objects:
        # softmax range
        logits_scores.append(sum(scores_dict_merged[o]) / len(scores_dict_merged[o]))
        vote_scores.append(len(scores_dict_merged[o]) * 1.0)
        assert len(scores_dict_merged[o]) <= 3

    # softmax the vote_scores
    
    logits_scores = torch.hstack(logits_scores)
    vote_scores = torch.tensor(vote_scores).softmax(-1)
    scores = 0.5 * logits_scores + 0.5 * vote_scores    

    # TODO: binary search for the threshold that returns the most objects below 5
    if dataset == "bird":
        if lm == "llama8" and embedding_model == "uae":
            threshold = -0.13
        elif lm == "llama8" and embedding_model == "snowflake":
            threshold = -0.07
        elif lm == "qwen7" and embedding_model == "uae":
            threshold = -100
        elif lm == "qwen7" and embedding_model == "snowflake":
            threshold = -100
    elif dataset == "ottqa":
        if lm == "llama8" and embedding_model == "uae":
            threshold = -0.019
        elif lm == "llama8" and embedding_model == "snowflake":
            threshold = -0.017
        elif lm == "qwen7" and embedding_model == "uae":
            threshold = -0.08
        elif lm == "qwen7" and embedding_model == "snowflake":
            threshold = -0.07
    elif dataset == "wikihop":
        threshold = -0.11
    
    top_idxs = torch.nonzero(scores >= threshold).squeeze(1)

    return [objects[idx] for idx in top_idxs]


def aggregate_votes(dataset, embedding_model: str, model: str, save: bool):
    '''Perform a weighted voting using logits and output the final set of retrieved objects'''

    fns = [
        f"./results/{dataset}/{embedding_model}_{model}/verify_aux_base_expand_{expand_k}_filtered.pkl" for expand_k in EXPAND_KS[dataset]
    ]

    preds_list = [read_pickle(fn) for fn in fns]

    if model == "llama8":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
    elif model == "qwen7":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    overall_preds = []

    qs = read_json(f"./data/{dataset}/dev.json")
    assert len(preds_list[0]) == len(qs)

    for q_idx, preds in enumerate(tqdm(zip(*preds_list), total=len(qs))):
        scores_dicts = []

        for pred in preds:
            if pred is None:
                scores_dicts.append(None)
                continue

            token_ids = pred["tokens"]
            (
                logits_normalized,
                logits_unnormalized,
                scores_normalized,
                scores_unnormalized,
            ) = pred["aux"]

            segments = get_object_segments(tokenizer, token_ids)
            scores_dict = get_object_score(
                token_ids, logits_normalized, segments, tokenizer
            )

            scores_dicts.append(scores_dict)

        if scores_dicts[0] is None or scores_dicts[0] == {}:
            overall_preds.append([])
        else:
            overall_preds.append(merge_scores(dataset, model, embedding_model, scores_dicts))

    if save:
        write_json(overall_preds, f"./results/{dataset}/{embedding_model}_{model}/pred.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["bird", "ottqa", "wikihop"])
    parser.add_argument("-embed", "--embedding_model", choices=["uae", "snowflake"])
    parser.add_argument("-lm", "--lm", choices=["llama8", "qwen7"])
    args = parser.parse_args()

    logger = ExperimentLogger.configure("aggregate", args.dataset, args.embedding_model, args.lm)
    logger.info("aggregate start")
    aggregate_votes(args.dataset, args.embedding_model, args.lm, save=True)
    logger.info("aggregate complete")
