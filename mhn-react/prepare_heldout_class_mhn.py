"""
Held-out-class data preparation for MHNreact.

Design (the "split-relabel" approach, chosen deliberately over regenerating
MHNreact's template vocabulary, since that would require depending on the
separate third-party template-relevance-master codebase whose relationship
to MHNreact's own template set is not confirmed):

  - MHNreact's existing 11,800-template vocabulary is kept as-is (it was
    already fixed once, from the full dataset, before any training run).
  - The held-out class's reactions are excluded entirely from train/valid,
    so the model receives literally zero training examples for that class's
    templates - equivalent to "training frequency = 0", a case already
    characterized elsewhere in this study.
  - Two self-contained CSVs are built per held-out class:
      prepro_heldout_class{N}_ood.csv.gz     (test = ALL of class N's reactions)
      prepro_heldout_class{N}_indist.csv.gz  (test = original test set minus class N)
    Both share identical train/valid content (class N excluded from both).

CRITICAL: every one of the original 50,016 rows must remain present in each
output file (relabeled, never dropped). MHNreact's template vocabulary is
built from ALL rows across ALL splits in the loaded file; dropping any row
silently shrinks that vocabulary and creates gaps in the label numbering,
which crashes training with a CUDA index-out-of-bounds error. Rows that
shouldn't participate in train/valid/test for a given run get tagged with
a distinct 'excluded' split value instead - present in the file (so their
templates still count toward the vocabulary), but ignored by training since
the training code only reads X['train']/X['valid']/X['test'] by name.

Run this on the MHNreact pod from /workspace/mhn-react.
"""
import pandas as pd

SRC = "data/USPTO_50k_MHN_prepro.csv.gz"
HELD_OUT_CLASSES = [4, 8]

df = pd.read_csv(SRC)
print(f"Loaded {SRC}: {len(df)} rows, split counts:")
print(df["split"].value_counts())
print(f"Total unique templates (whole file): {df['label'].nunique()}")

for held_out in HELD_OUT_CLASSES:
    is_class = df["class"] == held_out
    print(f"\n=== Class {held_out}: {is_class.sum()} total reactions "
          f"(train={((df['split']=='train') & is_class).sum()}, "
          f"valid={((df['split']=='valid') & is_class).sum()}, "
          f"test={((df['split']=='test') & is_class).sum()}) ===")

    # --- OOD file: class N's reactions become the test set. ---
    # Every row stays present; nothing is ever dropped.
    ood_df = df.copy()
    ood_df.loc[is_class, "split"] = "test"
    # Non-class-N rows that were originally 'test' can't also be 'test'
    # (that would contaminate the OOD-only test set) - relabel them
    # 'excluded' so they're still present (preserving template coverage)
    # but ignored by training/eval.
    orig_test_not_class = (df["split"] == "test") & (~is_class)
    ood_df.loc[orig_test_not_class, "split"] = "excluded"
    assert len(ood_df) == len(df), "row count changed - must never drop rows"
    assert ood_df["label"].nunique() == df["label"].nunique(), \
        "template vocabulary shrank - must preserve every row for full coverage"
    ood_path = f"data/USPTO_50k_MHN_prepro_heldout_class{held_out}_ood.csv.gz"
    ood_df.to_csv(ood_path, index=False, compression="gzip")
    print(f"  Saved {ood_path}: total={len(ood_df)}, "
          f"train={len(ood_df[ood_df['split']=='train'])}, "
          f"valid={len(ood_df[ood_df['split']=='valid'])}, "
          f"test={len(ood_df[ood_df['split']=='test'])}, "
          f"excluded={len(ood_df[ood_df['split']=='excluded'])}, "
          f"vocab={ood_df['label'].nunique()}")

    # --- In-distribution file: original test set minus class N. ---
    indist_df = df.copy()
    indist_df.loc[is_class, "split"] = "excluded"
    assert len(indist_df) == len(df), "row count changed - must never drop rows"
    assert indist_df["label"].nunique() == df["label"].nunique(), \
        "template vocabulary shrank - must preserve every row for full coverage"
    indist_path = f"data/USPTO_50k_MHN_prepro_heldout_class{held_out}_indist.csv.gz"
    indist_df.to_csv(indist_path, index=False, compression="gzip")
    print(f"  Saved {indist_path}: total={len(indist_df)}, "
          f"train={len(indist_df[indist_df['split']=='train'])}, "
          f"valid={len(indist_df[indist_df['split']=='valid'])}, "
          f"test={len(indist_df[indist_df['split']=='test'])}, "
          f"excluded={len(indist_df[indist_df['split']=='excluded'])}, "
          f"vocab={indist_df['label'].nunique()}")

print("\nDone. Both files preserve the full 50,016-row / 11,800-template "
      "vocabulary; only the 'split' tags differ.")
