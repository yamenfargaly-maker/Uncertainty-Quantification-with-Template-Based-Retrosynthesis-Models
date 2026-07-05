# Uncertainty Quantification and Calibration in Template-Based Retrosynthesis Models

**A Cross-Architecture Study of LocalRetro and MHNreact**

Yamen Fargaly, Princeton University
USPTO-50K, standard test split (n = 5,007)

Full paper: [paper/Uncertainty_Quantification_and_Calibration_in_Template_Based_Retrosynthesis_Models.pdf](paper/Uncertainty_Quantification_and_Calibration_in_Template_Based_Retrosynthesis_Models.pdf)

## Abstract

Template-based single-step retrosynthesis models output a softmax distribution over a template vocabulary, and their top score is routinely read as a confidence. This work asks whether that built-in signal reliably separates correct from incorrect predictions, and whether that separation survives distribution shift toward rare and novel templates. Across two architecturally distinct models on USPTO-50K (n = 5,007), LocalRetro (a graph neural network) and MHNreact (a Modern Hopfield retrieval network), the answer is no, and the failure takes two distinct forms. LocalRetro is well calibrated in aggregate but collapses sharply under template rarity; MHNreact is more uniformly miscalibrated. The standard perturbation remedies, MC dropout and deep ensembling, do not rescue the rare tail in either model, and in MHNreact both signals are inverted rather than merely uninformative. Per-tier (Mondrian) conformal selective prediction is the one method that restores error control. A held-out reaction-class experiment confirms the collapse under genuine novelty: with two entire reaction classes removed from training, top-1 accuracy falls to 0.00-0.25% while the model remains confident (out-of-distribution ECE up to 0.39).

## Key Findings

- **Baseline confidence discriminates, moderately, in both models.** AUROC = 0.710 (LocalRetro) and 0.738 (MHNreact) for separating correct vs. incorrect predictions using `1 - top_score`.
- **Calibration degrades with template rarity in both models, but with different failure signatures.** LocalRetro is well calibrated in-distribution (debiased ECE 0.024) and collapses sharply in the very-rare tier (0.291); MHNreact is mildly but uniformly miscalibrated across all tiers (0.038-0.107), with no sharp collapse. In LocalRetro's very-rare tier, confidence discrimination is statistically indistinguishable from random (AUROC 0.464).
- **MC dropout does not rescue the rare tail, and is actively inverted for MHNreact.** Dropout variance predicts a *correct* prediction more often than an incorrect one in MHNreact (AUROC 0.330), a result that holds under a boundary-robustness sweep.
- **Deep ensembling helps LocalRetro modestly, but not MHNreact.** A 4-seed ensemble improves overall LocalRetro AUROC (0.710 -> 0.727). MHNreact's ensemble disagreement is inverted in the very-rare tier (AUROC 0.427), replicated on a second, independent 4-seed ensemble.
- **Mondrian (per-tier) conformal selective prediction is the fix that works, and is architecture-agnostic.** A single global error threshold catastrophically fails the very-rare tier (keeps 9% of predictions at a 100% error rate); a per-tier threshold correctly abstains there instead.
- **A held-out-reaction-class experiment confirms genuine distribution-shift collapse.** On two reaction classes removed entirely from training, LocalRetro scores 0/909 and 0/811 correct, and MHNreact scores 1/909 and 2/811, while both models remain confidently wrong (OOD ECE 0.38-0.39 for LocalRetro, 0.26-0.27 for MHNreact).

## Repository Structure

```
LocalRetro/           Graph neural network retrosynthesis model (Chen & Jung, 2021)
  scripts/            Training, testing, MC dropout, TTA, calibration, conformal prediction,
                       deep ensemble, OOD-frequency, and reaction-center analysis code
  LocalTemplate/       Template extraction and decoding utilities
  preprocessing/       USPTO-50K preprocessing pipeline
  data/, models/       Datasets, extracted templates, and trained checkpoints
  outputs/             Predictions, calibration and analysis results

mhn-react/             Modern Hopfield Network retrosynthesis model (Seidl et al., 2022)
  mhnreact/            Core package: data, model, training, retrieval, evaluation
  scripts/             Training shell scripts (template relevance, single-step retrosynthesis)
  src/rdchiral/        Reaction template application/validation dependency
  data/                Processed USPTO-50K data, cached template embeddings, checkpoints
  mhn_*.py, *.sh        MC dropout, OOD evaluation, Mondrian conformal prediction, and
                       ensemble training/combination scripts used in the paper

paper/                 The full report (PDF) this README summarizes
```

## Methodology

Both models were trained from scratch on the standard USPTO-50K split (40,009 train / 5,002 validation / 5,007 test) with early stopping, and evaluated identically:

- **Baseline UQ:** `1 - top_score`, the softmax probability of the model's top-1 predicted template.
- **OOD proxy:** each test template's training-set frequency, bucketed into four tiers (rank-matched across models to control for LocalRetro's 671-template vs. MHNreact's 11,800-template vocabulary).
- **Perturbation UQ:** MC dropout (20 stochastic passes) and 4-seed deep ensembles.
- **Calibration:** reliability diagrams, expected/maximum calibration error (ECE/MCE) with a bias-floor correction for small-n tiers, and bootstrap/Bonferroni-corrected confidence intervals throughout.
- **Mitigation:** Mondrian (per-tier) conformal selective prediction, with a Hoeffding upper-confidence-bound correction for finite-sample optimism.
- **Genuine distribution shift:** two USPTO-50K reaction classes (Schneider et al. taxonomy) held out entirely from training and evaluated as an out-of-distribution test set.

See the [paper](paper/Uncertainty_Quantification_and_Calibration_in_Template_Based_Retrosynthesis_Models.pdf) for full methodology, tables, and figures.

## References

- Chen, S. & Jung, Y. "Deep Retrosynthetic Reaction Prediction using Local Reactivity and Global Attention." *JACS Au*, 2021.
- Seidl, P. et al. "Improving Few- and Zero-Shot Reaction Template Prediction Using Modern Hopfield Networks." *J. Chem. Inf. Model.*, 2022.
