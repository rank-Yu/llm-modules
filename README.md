<div align="center">
  <h2><b> [AAAI'26 Oral] Discovering Decoupled Functional Modules in Large Language Models </b></h2>
</div>

This repository contains the official implementation for our **AAAI 2026 Oral** paper: Discovering Decoupled Functional Modules in Large Language Models.

> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚

<!--
```bibtex
```
-->

## Abstract

Understanding the internal functional organization of Large Language Models (LLMs) is crucial for improving their trustworthiness and performance.
However, how LLMs organize different functions into modules remains highly unexplored.
To bridge this gap, we formulate a function module discovery problem and propose an Unsupervised LLM Cross-layer MOdule Discovery (ULCMOD) framework that simultaneously disentangles the large set of neurons in the entire LLM into modules while discovering the topics of input samples related to these modules.
Our framework introduces a novel objective function and an efficient Iterative Decoupling (IterD) algorithm.
Extensive experiments show that our method discovers high-quality, disentangled modules that capture more meaningful semantic information and achieve superior performance in various downstream tasks.
Moreover, our qualitative analysis reveals that the discovered modules show function comprehensiveness, function hierarchy, and clear function spatial arrangement within LLMs.
Our work provides a novel tool for interpreting LLMs' function modules, filling a critical gap in LLMs' interpretability research.

## Dependencies

We test the codebase with Python 3.10+. Creating an isolated environment before installing the requirements avoids version conflicts:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Copy the environment template and point `LLM_PATH_BASE` to the directory that hosts your local LLM weights (e.g., the root folder downloaded from Hugging Face):

    ```bash
    cp .env.example .env
    ```

    Open `.env` and update the variable so every script can resolve the LLM checkpoints.

2. (Optional) Adjust `run.sh` if you want to change the LLM name, GPU id, number of clusters, or which stages to execute. The script currently runs `p1-p6` sequentially for `Qwen2.5-3B-Instruct`.

3. Launch the full pipeline:

    ```bash
    bash run.sh
    ```

    This runs dataset generation (`p1_generate_dataset.py`), activation extraction (`p2_get_activation.py`), clustering (`p3_clustering.py`), iterative decoupling (`p4_iter.py`), $\mathcal{L}$, $\xi$ and $B$ reporting (`p5_result_objective.py`), and generalization performance evaluation (`p6_result_prediction.py`).
