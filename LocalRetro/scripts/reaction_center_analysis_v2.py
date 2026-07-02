"""
reaction_center_analysis_v2.py

Uses the AUTHORITATIVE training labels straight from preprocessed_test.csv
(the 'Labels' column, e.g. "[('b', 13, 468)]") instead of reconstructing
the reaction core from atom maps. This guarantees the ground truth is
exactly what the model was trained against -- no approximation, no
indexing-assumption risk.

For each molecule, compares the model's top-1 raw prediction (type, index,
template) against the label set in two ways:
  - exact_match:    (type, index) exactly matches one of the ground-truth
                     edit labels for this reaction
  - template_match: the predicted template_id matches one of the
                     ground-truth labels' template_id, even if the index
                     differs (catches symmetry-equivalent sites -- e.g.
                     predicting bond 20 vs the canonical label's bond 13,
                     where both are chemically equivalent positions that
                     the same template could apply to)

Cross-tabulates both against the existing 'error' column (correct/incorrect
top-1 prediction) from localretro_baseline.csv.

Inputs:
  - ../data/USPTO_50K/preprocessed_test.csv
  - ../outputs/raw_prediction/LocalRetro_USPTO_50K.txt
  - ../outputs/localretro_baseline.csv

Run from scripts/ directory:
    python reaction_center_analysis_v2.py
"""

import argparse
import ast
import csv
import re


def load_labels(path):
    """
    Returns dict: mol_idx (int, 0-indexed row order) -> list of (type, index, template_id) tuples
    """
    labels = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for mol_idx, row in enumerate(reader):
            raw = row['Labels']
            try:
                parsed = ast.literal_eval(raw)
                # parsed is a list of tuples like [('b', 13, 468)]
                labels[mol_idx] = [(t, int(i), int(tpl)) for (t, i, tpl) in parsed]
            except Exception:
                labels[mol_idx] = []
    return labels


def parse_raw_prediction_top1(line):
    parts = line.rstrip('\n').split('\t')
    test_id = int(parts[0])
    top1_str = parts[2].strip()
    m = re.match(r'\(([ab]), (\d+), (\d+), ([\d.]+)\)', top1_str)
    if not m:
        return test_id, None, None, None, None
    edit_type = m.group(1)
    site_idx = int(m.group(2))
    template_id = int(m.group(3))
    score = float(m.group(4))
    return test_id, edit_type, site_idx, template_id, score


def load_errors(path):
    errors = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors[int(row['mol_idx'])] = int(row['error'])
    return errors


def main(args):
    print('Loading authoritative training labels from %s ...' % args.preprocessed)
    labels = load_labels(args.preprocessed)
    print('Loaded labels for %d molecules.' % len(labels))

    print('Loading baseline error labels...')
    errors = load_errors(args.baseline)

    print('Processing raw predictions...')
    results = []
    n_skipped = 0

    with open(args.raw_prediction, 'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line == '':
                continue
            test_id, edit_type, site_idx, template_id, score = parse_raw_prediction_top1(line)
            if edit_type is None or test_id not in labels:
                n_skipped += 1
                continue

            label_set = labels[test_id]
            exact_match = any(t == edit_type and i == site_idx for (t, i, tpl) in label_set)
            template_match = any(tpl == template_id for (t, i, tpl) in label_set)

            error = errors.get(test_id, None)
            results.append({
                'mol_idx': test_id,
                'edit_type': edit_type,
                'site_idx': site_idx,
                'template_id': template_id,
                'exact_match': int(exact_match),
                'template_match': int(template_match),
                'error': error,
            })

    print('Processed %d molecules (%d skipped).' % (len(results), n_skipped))

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'mol_idx', 'edit_type', 'site_idx', 'template_id', 'exact_match', 'template_match', 'error'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print('Wrote %s' % args.output)

    valid = [r for r in results if r['error'] is not None]
    correct = [r for r in valid if r['error'] == 0]
    wrong = [r for r in valid if r['error'] == 1]

    def rate(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float('nan')

    print()
    print('=== Overall ===')
    print('exact_match rate (all):      %.4f' % rate(valid, 'exact_match'))
    print('template_match rate (all):   %.4f' % rate(valid, 'template_match'))

    print()
    print('=== SANITY CHECK: among CORRECT predictions (error=0) ===')
    print('exact_match rate:    %.4f (%d / %d)' % (rate(correct, 'exact_match'), sum(r['exact_match'] for r in correct), len(correct)))
    print('template_match rate: %.4f (%d / %d)  <- should be very close to 1.0; if not, top-1 reactant correctness and template_match are diverging unexpectedly' % (
        rate(correct, 'template_match'), sum(r['template_match'] for r in correct), len(correct)))

    correct_a = [r for r in correct if r['edit_type'] == 'a']
    correct_b = [r for r in correct if r['edit_type'] == 'b']
    if correct_a:
        print('  -> atom-type, correct only: exact=%.4f  template=%.4f  (n=%d)' % (
            rate(correct_a, 'exact_match'), rate(correct_a, 'template_match'), len(correct_a)))
    if correct_b:
        print('  -> bond-type, correct only: exact=%.4f  template=%.4f  (n=%d)' % (
            rate(correct_b, 'exact_match'), rate(correct_b, 'template_match'), len(correct_b)))

    print()
    print('=== Among INCORRECT predictions (error=1) ===')
    print('exact_match rate:    %.4f' % rate(wrong, 'exact_match'))
    print('template_match rate: %.4f' % rate(wrong, 'template_match'))

    print()
    print('=== Cross-tab among CORRECT predictions: "right answer, right SITE" vs "right answer, right TEMPLATE but different site" vs "right answer, neither" ===')
    n_exact = sum(1 for r in correct if r['exact_match'] == 1)
    n_template_only = sum(1 for r in correct if r['exact_match'] == 0 and r['template_match'] == 1)
    n_neither = sum(1 for r in correct if r['exact_match'] == 0 and r['template_match'] == 0)
    total_c = len(correct)
    print('Exact site + template match:               %d (%.2f%%)' % (n_exact, 100 * n_exact / total_c))
    print('Same template, different (symmetric) site:  %d (%.2f%%)' % (n_template_only, 100 * n_template_only / total_c))
    print('Neither matches (should be ~0 if correct):   %d (%.2f%%)' % (n_neither, 100 * n_neither / total_c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reaction center analysis v2 -- uses authoritative training labels')
    parser.add_argument('--preprocessed', default='../data/USPTO_50K/preprocessed_test.csv')
    parser.add_argument('--raw_prediction', default='../outputs/raw_prediction/LocalRetro_USPTO_50K.txt')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--output', default='../outputs/reaction_center_analysis_v2.csv')
    args = parser.parse_args()
    main(args)
