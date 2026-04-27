# Step 1: information alignment

import numpy as np
from tiger_utils import read_json, write_json, merge, concat
from tiger_utils.model import user, assistant, system
from nltk.stem import PorterStemmer
from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
import argparse
from bm25s import BM25

from utils.execute import run_prompts
from utils.utils import chunk_id_to_original_id
from metrics import get_gold_objects

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def obtain_keywords(
    lm: str, embedding_model: str, dataset: str, num_partitions, partition: int
):
    """
    Use LLM to constrain decode relevant keywords that appear in the corpus
    """

    examples = read_json(f"./data/{dataset}/examples.json")
    assert len(examples) == 3

    base_prompt = [
        system(
            "You are given a user question, your task is to decompose the user question into contiguous, non-overlapping substrings that can cover different information mentioned in the user question. For each substring, generate words that are the most relevant to the substring. You should end your response with <> when you finish generating relevant words for each substring."
        ),
    ]

    for example in examples:
        base_prompt.append(user(f"User question: {example['question']}"))
        resp = f"Keywords: {example['keywords']} #sep# Similar words: {example['similar words']} <>"
        base_prompt.append(assistant(resp))

    qs = read_json(f"./data/{dataset}/dev.json")

    q_idxs, prompts = [], []

    for q_idx, q in enumerate(qs):
        question = q["question"]
        prompt = base_prompt + [
            user(f"User question: {question.lower().replace('(', '').replace(')', '')}")
        ]
        prompts.append(prompt)
        q_idxs.append(q_idx)

    run_prompts(
        num_partitions,
        partition,
        prompts,
        f"./results/{dataset}/{embedding_model}_{lm}/ia",
        lm,
        embedding_model,
        q_idxs=q_idxs,
        dataset=dataset,
    )

import re
from nltk.stem import PorterStemmer
import logging

def extract_word_from_parentheses(s: str):
    pred_keywords_tmp = re.findall(r"\((.*?)\)", s)
    pred_keywords = set()
    for keywords_tmp in pred_keywords_tmp:
        pred_keywords.update(re.split(", | ", keywords_tmp))
    pred_keywords = pred_keywords - {"", " "}
    return list(pred_keywords)


def extract_keywords(s: str):
    s = s.replace("Keywords: ", "").strip()
    s = s.split(" | ")
    s = concat([x.split(" ") for x in s])
    s = list(set(s))
    return s


def lookup_objects_keywords(s: str, uae_objects, vocab_dict, k, bm25=None):
    stemmer = PorterStemmer()

    pred_keywords = extract_word_from_parentheses(s)
    pred_keywords = [stemmer.stem(x.lower()) for x in pred_keywords]

    logger = logging.getLogger("bm25s")
    logger.setLevel(logging.INFO)

    if len(pred_keywords) == 0:
        pred_objects = []
    else:
        bm25_objects, bm25_scores = bm25.retrieve(
            [pred_keywords],
            corpus=list(vocab_dict.keys()),
            k=len(vocab_dict),
            show_progress=False,
            sorted=True,
        )
        bm25_objects, bm25_scores = bm25_objects.tolist()[0], bm25_scores[0]
        bm25_scores /= bm25_scores.max()
        bm25_score_dict = {}
        for o, s in zip(bm25_objects, bm25_scores):
            bm25_score_dict[o] = s

        pred_scores = []
        for o, uae_score in zip(uae_objects[0], uae_objects[1]):
            pred_scores.append(0.9 * uae_score + 0.1 * bm25_score_dict[o])
        pred_scores = np.array(pred_scores)

        top_idxs = np.argsort(-pred_scores)[:k]
        pred_objects = [uae_objects[0][idx] for idx in top_idxs]

    return pred_objects


def obtain_base_search_objects(
    lm: str, 
    embedding_model: str, 
    dataset: str, 
    save: bool
    ):
    """
    Obtain the base search objects by doing a hybrid search using dense retrievers and keywords from `obtain_keywords`
    """

    qs = read_json(f"./data/{dataset}/dev.json")

    stemmer = PorterStemmer()

    vocab_dict = read_json(f"./data/{dataset}/vocab_dev.json")

    pred_data = read_json(f"./results/{dataset}/{embedding_model}_{lm}/ia.json")

    dense_top_objects = read_json(
        f"./data/{dataset}/embeds/{embedding_model}_pred_dev.json"
    )
    for q_idx in range(len(dense_top_objects)):
        dense_top_objects[q_idx][0] = [
            chunk_id_to_original_id(dataset, x) for x in dense_top_objects[q_idx][0]
        ]

    vocab_dict = {
        o: list(set(stemmer.stem(x) for x in vocab_dict[o])) for o in tqdm(vocab_dict)
    }

    if dataset == "wikihop":
        bm25 = BM25(b=0)
    else:
        bm25 = BM25()

    bm25.index(list(vocab_dict.values()), show_progress=False)

    preds, preds_full, golds = [], [], []

    non_overlapping_cnt = 0

    skip_idxs = []

    for q_idx, q in enumerate(tqdm(qs)):
        pred_text_raw = pred_data[q_idx]

        gold_objects = get_gold_objects(dataset, q)
        if len(gold_objects) == 1 and dataset == "ottqa":
            preds_full.append([])
            continue

        if pred_text_raw.count("#sep#") < 1 or "<>" not in pred_text_raw:
            skip_idxs.append(q_idx)
            preds_full.append([])
            continue

        pred_objects = lookup_objects_keywords(
            pred_text_raw.split(" #sep# ")[1],
            dense_top_objects[q_idx],
            vocab_dict,
            k=5,
            bm25=bm25,
        )

        preds_full.append(deepcopy(pred_objects))
        pred_objects = [x.upper() for x in pred_objects]
        preds.append(deepcopy(pred_objects))

        gold_objects = [x.upper() for x in gold_objects]
        golds.append(gold_objects)

        if len(set(pred_objects) & set(gold_objects)) == 0:
            non_overlapping_cnt += 1

    if save:
        write_json(preds_full, f"./results/{dataset}/{embedding_model}_{lm}/base.json")


if __name__ == "__main__":
    set_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--partition", type=int)
    parser.add_argument("-d", "--dataset", choices=["bird", "ottqa", "wikihop"])
    parser.add_argument("-embed", "--embedding_model", choices=["uae", "snowflake"])
    parser.add_argument("-lm", "--lm", choices=["llama8", "qwen7"])
    args = parser.parse_args()

    num_partitions = 8
    # Step 1: Execute to get the output first (multi-process, 0 to num_partitions-1)

    # 直接将target设为obtain_keywords即可，无需额外包装函数
    if args.partition is None:
        #single process
        for partition in range(num_partitions):
            logger.info("Running IA for partition %d/%d...", partition + 1, num_partitions)
            obtain_keywords(
                args.lm, args.embedding_model, args.dataset, 
                num_partitions, partition=partition
            )

        # Step 2: Merge the outputs
        logger.info("Merging outputs...")
        merge(num_partitions, f'./results/{args.dataset}/{args.embedding_model}_{args.lm}/ia', 'json')
        logger.info("Merging complete.")
        # Step 3: Parse the outputs to get the base objects
        obtain_base_search_objects(args.lm, args.embedding_model, args.dataset, save=True)
        logging.info("Base search objects obtained and saved.")
        
    else:
        obtain_keywords(
            args.lm, args.embedding_model, args.dataset, num_partitions, args.partition
        )
    # All processes complete, now gather/merge as needed.
