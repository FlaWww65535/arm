# 💪 ARM: Alignment-oriented retrieval method

If you find our code, or the paper helpful, please cite the paper

```
@article{chen2025can,
  title={Can we retrieve everything all at once? ARM: An alignment-oriented LLM-based retrieval method},
  author={Chen, Peter Baile and Zhang, Yi and Cafarella, Michael and Roth, Dan},
  journal={arXiv preprint arXiv:2501.18539},
  year={2025}
}
```

## Overview
1. Information alignment: `align_info.py`
2. Structural alignment
    1. Expand the base search objects obtained from information alignment: `align_structure_expand.py`
    2. Filter the expanded search objects into drafts: `align_structure_filter.py`
3. Self-verification and aggregation
    1. Self-verification: `verify.py`
    2. Aggregation: `aggregate.py`

## Setup
Download the [data](https://drive.google.com/file/d/1wyPFmj_SO-iOm_Yh2RNe3cIxO4Jp8DT3/view?usp=drive_link) folder and put it in the root directory

The code can be executed in a pipelined fashion in the order mentioned in [Overview](#overview), so
```
python align_info.py --> python align_structure_expand.py --> ...
```

Files with
```python
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--partition', type=int)
args = parser.parse_args()
```
support parallel execution. `num_partitions` specifies the the number of jobs executed in parallel. To run these files (e.g., `align_info.py`), you can use, for example,
```
python align_info.py -p 0 & python align_info.py -p 1 & ...
```
followed by `merge(...)` in `align_info.py` to merge the outputs


## Contact
If you have any questions or feedback, please send an email to peterbc@mit.edu.