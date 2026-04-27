# ARM 最简试验运行流程

本文基于 `arm/README.md` 和当前代码入口整理。所有命令默认从 `arm/` 目录执行，因为代码大量使用 `./data`、`./results` 相对路径。

## 0. 准备

```bash
cd /home/lyh/src/MTR/arm
```

下载 README 中的数据包，解压后放到：

```text
arm/data/
```

安装 Python 依赖：

```bash
pip install -r requirements.txt
```

代码还依赖作者的 `tiger_utils` 包；该包未随仓库提供，需要另行安装，或把它放到 `PYTHONPATH`。使用 `llama8` 或 `qwen7` 时，还需要可访问对应 HuggingFace 模型：

- `llama8`: `meta-llama/Llama-3.1-8B-Instruct`
- `qwen7`: `Qwen/Qwen2.5-7B-Instruct`

下面命令中的 `xargs -P` 是并行度；GPU/显存不足时调小即可。

## 1. 选择试验参数

```bash
export DATASET=ottqa        # 可选: bird / ottqa / wikihop
export EMBED=uae            # 可选: uae / snowflake
export LM=llama8            # 可选: llama8 / qwen7
```

按数据集设置结构扩展分片数和扩展 k：

```bash
case "$DATASET" in
  bird)    export EXPAND_PARTS=10; export KS="2 3 4" ;;
  ottqa)   export EXPAND_PARTS=40; export KS="3 4 5" ;;
  wikihop) export EXPAND_PARTS=20; export KS="1 2 3" ;;
esac
```

## 2. 信息对齐

并行生成关键词：

```bash
seq 0 7 | xargs -I{} -P 8 python align_info.py -p {} -d "$DATASET" -embed "$EMBED" -lm "$LM"
```

合并分片，并生成基础检索对象：

```bash
python -c "from tiger_utils import merge; merge(8, './results/${DATASET}/${EMBED}_${LM}/ia', 'json')"
python -c "from align_info import obtain_base_search_objects; obtain_base_search_objects('${LM}', '${EMBED}', '${DATASET}', save=True)"
```

注意：`align_info.py` 输出 `./results/.../base.json`，但 `align_structure_expand.py` 读取 `./keyword_objects/.../base.json`。保持源码不变时，复制一次：

```bash
mkdir -p "./keyword_objects/${DATASET}/${EMBED}_${LM}"
cp "./results/${DATASET}/${EMBED}_${LM}/base.json" "./keyword_objects/${DATASET}/${EMBED}_${LM}/base.json"
```

## 3. 结构对齐

扩展基础检索对象：

```bash
seq 0 $((EXPAND_PARTS - 1)) | xargs -I{} -P "$EXPAND_PARTS" python align_structure_expand.py -p {} -d "$DATASET" -embed "$EMBED" -lm "$LM"
```

合并每个 `expand_k` 的扩展结果：

```bash
for k in $KS; do
  python -c "from tiger_utils import merge; merge(${EXPAND_PARTS}, './results/${DATASET}/${EMBED}_${LM}/base_expand_${k}', 'json')"
done
```

用 MIP/ILP 过滤扩展对象：

```bash
seq 0 9 | xargs -I{} -P 10 python align_structure_filter.py -p {} -d "$DATASET" -embed "$EMBED" -lm "$LM"
```

合并过滤结果：

```bash
for k in $KS; do
  python -c "from tiger_utils import merge; merge(10, './results/${DATASET}/${EMBED}_${LM}/base_expand_${k}_filtered', 'json')"
done
```

## 4. 自验证与聚合

对每个 `expand_k` 做 LLM 自验证：

```bash
for k in $KS; do
  seq 0 7 | xargs -I{} -P 8 python verify.py -p {} -k "$k" -d "$DATASET" -embed "$EMBED" -lm "$LM"
done
```

合并验证输出：

```bash
for k in $KS; do
  python -c "from tiger_utils import merge; merge(8, './results/${DATASET}/${EMBED}_${LM}/verify_base_expand_${k}_filtered', 'json')"
  python -c "from tiger_utils import merge; merge(8, './results/${DATASET}/${EMBED}_${LM}/verify_aux_base_expand_${k}_filtered', 'pkl')"
done
```

聚合投票，生成最终检索结果：

```bash
python aggregate.py -d "$DATASET" -embed "$EMBED" -lm "$LM"
```

最终产物：

```text
./results/${DATASET}/${EMBED}_${LM}/pred.json
```

## 5. 可选评测

检索评测：

```bash
python -c "from tiger_utils import read_json; from metrics import eval_retrieval; eval_retrieval('${DATASET}', read_json('./results/${DATASET}/${EMBED}_${LM}/pred.json'))"
```

## 6. 流程总览

```text
align_info.py
  -> merge ia
  -> obtain_base_search_objects
  -> align_structure_expand.py
  -> merge base_expand_k
  -> align_structure_filter.py
  -> merge base_expand_k_filtered
  -> verify.py
  -> merge verify / verify_aux
  -> aggregate.py
  -> results/.../pred.json
```
