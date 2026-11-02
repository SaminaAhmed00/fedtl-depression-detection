# Federated Transfer Learning for Depression Detection across Disjoint EEG, Speech, and Text Silos

Code and results for the paper

> S. Ahmed, S. K. Saritha, B. Roy. *Federated Transfer Learning for Depression Detection across Disjoint EEG, Speech, and Text Silos: Benefits, Limits, and Evaluation Risks.* Submitted to *Biomedical Signal Processing and Control*, 2026.

Three clinical datasets that share no patients, no feature space and no severity instrument are treated as three
institutional silos:

| Silo | Dataset | Subjects | Instrument | Frozen encoder | Embedding |
|------|---------|---------:|------------|----------------|----------:|
| EEG | MODMA (3-channel resting state) | 55 | PHQ-9 | MSCA-CNN-GRU (trained inside each partition) | 128 |
| Speech | DAIC-WOZ | 189 | PHQ-8 | WavLM-base-plus, layer ℓ, mean ⊕ SD of turns | 1536 |
| Text | PDCH (Chinese consultations) | 99 | HAMD-17 | RoBERTa-wwm-ext, mean over tokens | 768 |

Severity labels are harmonized onto three common bands. Each silo trains a local projection head and classifier,
and silos exchange **only class prototypes** with a server (FedProto-style or FedTGP-style aggregation, simplex-ETF
initialization). Every configuration is evaluated over 10 subject-level partitions × 3 seeds against a
**matched local control** that differs only in the absence of cross-silo exchange.

**Main findings:** cross-silo prototype exchange gave no significant gain on any endpoint, and on MODMA the same EEG
model scored 74.3% window-level accuracy under segment-wise splitting but 49.6% (chance) under subject-wise splitting.

---

## Repository layout

```
notebooks/
  1_preprocessing/
    eeg/     1_MODMA_EEG_Preprocessing      filtering, 4-s windows, artifact handling
             2_MODMA_EEG_Final_Best         MSCA-CNN-GRU encoder (development version)
             3_MODMA_EEG_Proposed           encoder variants explored during development
             4_MODMA_VMD_Stability_Test     VMD feature stability check
    audio/   1_DAICWOZ_audio_embeddings     WavLM layer probing and turn-level embeddings
    text/    1_PDCH_text_preprocessing      transcript parsing (patient/doctor turns)
             2_PDCH_text_embeddings         RoBERTa-wwm-ext embeddings used in the paper
             3_PDCH_text_finetune           fine-tuning / LoRA comparison (development)
             4_PDCH_text_gentle             light-touch fine-tuning comparison (development)
  2_label_harmonization/
             0_FL_Label_Harmonization       PHQ-9 / PHQ-8 / HAMD-17 -> 3 severity bands
  3_experiments/                            <- everything reported in the paper
             1_make_partitions              10 stratified subject-level partitions (60/20/20)
             2_EEG_outoffold_embeddings_and_leakage
                                            leakage-free EEG embeddings + segment- vs subject-wise comparison
             3_FL_experiments               all federated runs, statistics, LaTeX tables (single session)
             3a_/3b_FL_experiments_session  the same notebook split into two parallel Colab sessions
  4_development/                            earlier single-split pipeline, ablations and plots (kept for transparency)
src/
  stats.py                                  corrected resampled t-test (Nadeau–Bengio), Holm, Wilcoxon, d_z
results/                                    CSV outputs of the experiments
  tables/                                   LaTeX tables and macros used in the manuscript
figures/                                    manuscript figures
```

## Data access

The datasets are **not** included and cannot be redistributed. Request them from their custodians:

- **MODMA** — Cai et al., *Scientific Data* 9:178 (2022). Request access at <https://modma.lzu.edu.cn/data/index/>.
- **DAIC-WOZ** — Gratch et al., LREC 2014. Request access at <https://dcapswoz.ict.usc.edu/>.
- **PDCH** — Cao et al., *Scientific Data* 12:1577 (2025). Follow the access instructions in the paper.

The notebooks were run on Google Colab and expect the data on Google Drive:

```
/content/drive/MyDrive/Depression_FL_Project/
  data/raw/MODMA/          MODMA 3-channel EEG + subjects_information_EEG_*.xlsx
  data/raw/PDCH/           PDCH transcripts and HAMD-17 scores
  data/processed/eeg/      written by the EEG preprocessing notebook
  data/embeddings/         frozen audio/text embeddings written by the preprocessing notebooks
  fl/revision_r1/          all experiment outputs (partitions, runs, tables)
```

Change `BASE` / the Drive paths at the top of each notebook if your layout differs.

## How to reproduce the paper

1. **Preprocess and embed** each modality: run the notebooks in `1_preprocessing/eeg`, `audio` (GPU) and
   `text` (GPU, notebook 1 then 2).
2. **Harmonize labels:** `2_label_harmonization/0_FL_Label_Harmonization.ipynb`.
3. **Experiments** (`3_experiments/`, in order):

| Step | Notebook | GPU | Output |
|------|----------|-----|--------|
| 1 | `1_make_partitions` | no | `partitions.csv`, split-count table |
| 2 | `2_EEG_outoffold_embeddings_and_leakage` | yes (A100 recommended) | per-partition EEG embeddings, leakage comparison, `results_macros_eeg.tex` |
| 3 | `3_FL_experiments` **or** `3a` + `3b` in two parallel sessions | yes (L4/T4 fine) | `runs_long.csv`, `executed_config.json`, all tables and statistics |

Step 3 is roughly 1,000 training runs. It saves after every run, so a disconnected Colab session can simply be
re-run and it resumes where it stopped. All random streams are seeded per run; a self-test trains one
configuration twice and requires identical outputs.

4. Copy `fl/revision_r1/latex/*` into `results/tables/` to regenerate the manuscript tables.

## Requirements

Python 3.10+, PyTorch 2.x with CUDA. See `requirements.txt`; on Colab most packages are pre-installed.

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the paper (citation will be updated on publication):

```bibtex
@article{ahmed2026federated,
  title   = {Federated Transfer Learning for Depression Detection across Disjoint EEG, Speech, and Text Silos: Benefits, Limits, and Evaluation Risks},
  author  = {Ahmed, Samina and Saritha, Sri Khetwat and Roy, Bholanath},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see `LICENSE`). The datasets remain subject to their own licence agreements.

## Contact

Samina Ahmed — saminaahmed2405@gmail.com
Centre of Artificial Intelligence, Maulana Azad National Institute of Technology (MANIT), Bhopal, India
