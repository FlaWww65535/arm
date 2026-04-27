import os
import logging
from functools import partial
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from tiger_utils import read_json, write_json, read_pickle, write_pickle

from utils.utils import Execute, create_directory
from constraint_decoder import ConstraintDecoder, prefix_allowed_tokens_fn
from constraint_decoder_rerank import ConstraintDecoderRerank

log_dir = Path("results")
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.FileHandler(
    log_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}.txt", encoding="utf-8"
)
handler.setLevel(logging.DEBUG)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

def run_prompts(
    num_partitions,
    partition: int,
    prompts,
    _fn: str,
    model: str,
    embedding_model: str|None = None,
    dataset=None,
    q_idxs: list[int]|None = None,
    stage: str|None = None,
    ilp_fn=None,
    return_logits=False,
):
    if num_partitions is not None:
        interval = (len(prompts) // num_partitions) + 1
        start, end = partition * interval, (partition + 1) * interval
        prompts = prompts[start:end]

    in_constraint_mode = q_idxs is not None

    if in_constraint_mode and num_partitions is not None:
        q_idxs = q_idxs[start:end]

    if ilp_fn:
        fn, aux_fn = (
            f'{_fn}_{ilp_fn.replace("ilp_", "")}_{partition}.json',
            f'{_fn}_aux_{ilp_fn.replace("ilp_", "")}_{partition}.pkl',
        )
    else:
        fn, aux_fn = f"{_fn}_{partition}.json", f"{_fn}_aux_{partition}.pkl"

    if num_partitions is not None:
        if return_logits:
            print(partition, start, end, fn, aux_fn)
        else:
            print(partition, start, end, fn)

    create_directory(fn)

    outputs, aux_outputs = [], []
    if os.path.isfile(fn):
        outputs = read_json(fn)
    if os.path.isfile(aux_fn):
        aux_outputs = read_pickle(aux_fn)

    exec = Execute(model)

    if in_constraint_mode:
        if stage == "rerank":
            constraint_decoder = ConstraintDecoderRerank(
                tokenizer=exec.model["tokenizer"],
                dataset=dataset,
                ilp_fn=ilp_fn,
                model_name=model,
                embedding_model=embedding_model,
            )
        else:
            constraint_decoder = ConstraintDecoder(
                tokenizer=exec.model["tokenizer"],
                model=exec.model["model"],
                dataset=dataset,
                model_name=model,
                embedding_model=embedding_model,
            )
        constraint_decoder.reset(q_idxs[0])
        partial_prefix_allowed_tokens_fn = partial(
            prefix_allowed_tokens_fn, constraint_decoder=constraint_decoder
        )

    for idx, prompt in enumerate(tqdm(prompts)):
        if idx < len(outputs):
            continue

        try:
            if prompt is None:
                output, aux_output = "", None
            else:
                if in_constraint_mode:
                    constraint_decoder.reset(q_idxs[idx])
                    output = exec.inference(
                        p=prompt,
                        partial_prefix_allowed_tokens_fn=partial_prefix_allowed_tokens_fn,
                        return_logits=return_logits,
                    )
                    if return_logits:
                        aux_output = {"tokens": output["tokens"], "aux": output["aux"]}
                        output = output["sentence"]
                else:
                    output = exec.inference(p=prompt)
        except Exception as e:
            print(e)
            logger.exception("Inference failed")
            output, aux_output = "", None

        outputs.append(output)

        if _fn is not None:
            write_json(outputs, fn)

        if return_logits:
            aux_outputs.append(aux_output)
            write_pickle(aux_outputs, aux_fn)
