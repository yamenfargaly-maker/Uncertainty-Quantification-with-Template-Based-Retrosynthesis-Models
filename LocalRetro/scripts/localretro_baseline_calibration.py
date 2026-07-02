"""
localretro_baseline_calibration.py

Builds the per-molecule UQ table for the LocalRetro baseline.

Inputs:
  - ../outputs/decoded_prediction/LocalRetro_USPTO_50K.txt
        format: test_id \t ('reactant_smiles', score) \t ('reactant_smiles', score) \t ...
  - ../data/USPTO_50K/raw_test.csv
        format: id,class,reactants>reagents>production   (atom-mapped)

Output:
  - ../outputs/localretro_baseline.csv
        columns: mol_idx, top_score, uq, pred_top1, true_reactants, error

Run from the scripts/ directory:
    python localretro_baseline_calibration.py
"""

import ast
import re
import csv
import argparse

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def strip_atom_map(smiles):
    """Remove atom-map numbers (e.g. [CH2:5] -> [CH2]) before canonicalizing."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical_smiles(smiles):
    """Canonicalize a (possibly multi-fragment, possibly atom-mapped) SMILES string."""
    if smiles is None or smiles == '':
        return None
    mol = strip_atom_map(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def canonical_reactant_set(smiles):
    """
    Canonicalize each fragment of a (possibly multi-component, '.'-separated)
    reactant SMILES independently, then return as a sorted tuple so that
    fragment order doesn't matter for comparison.
    """
    if smiles is None or smiles == '':
        return None
    fragments = smiles.split('.')
    canon_fragments = []
    for frag in fragments:
        c = canonical_smiles(frag)
        if c is None:
            return None  # if any fragment fails to parse, treat whole thing as unparseable
        canon_fragments.append(c)
    return tuple(sorted(canon_fragments))


def load_raw_test(path):
    """
    Returns dict: mol_idx (int, 0-indexed row order) -> true_reactants_smiles (raw, atom-mapped)
    """
    ground_truth = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # find the reaction column (handles both 'id,class,reactants>reagents>production'
        # and bare 'reactants>reagents>production' formats)
        rxn_col_idx = None
        for i, col in enumerate(header):
            if 'reactants' in col and 'production' in col:
                rxn_col_idx = i
                break
        if rxn_col_idx is None:
            raise ValueError('Could not find reactants>reagents>production column in %s. Header was: %s' % (path, header))

        for mol_idx, row in enumerate(reader):
            rxn = row[rxn_col_idx]
            reactants_part = rxn.split('>')[0]
            ground_truth[mol_idx] = reactants_part
    return ground_truth


def parse_decoded_line(line):
    """
    Parses one line of the decoded_prediction file:
        test_id \t ('smiles1', score1) \t ('smiles2', score2) \t ...
    Returns (test_id, list of (smiles, score) tuples)
    """
    parts = line.rstrip('\n').split('\t')
    test_id = int(parts[0])
    preds = []
    for p in parts[1:]:
        p = p.strip()
        if p == '':
            continue
        try:
            smiles, score = ast.literal_eval(p)
            preds.append((smiles, float(score)))
        except Exception:
            continue
    return test_id, preds


def main(args):
    print('Loading ground truth from %s ...' % args.raw_test)
    ground_truth = load_raw_test(args.raw_test)
    print('Loaded %d ground-truth reactions.' % len(ground_truth))

    print('Loading decoded predictions from %s ...' % args.decoded)
    rows_out = []
    n_total = 0
    n_error = 0
    n_unparseable_pred = 0
    n_unparseable_truth = 0

    with open(args.decoded, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            test_id, preds = parse_decoded_line(line)
            if test_id not in ground_truth:
                continue
            if len(preds) == 0:
                continue

            top1_smiles, top_score = preds[0]
            uq = 1.0 - top_score

            true_raw = ground_truth[test_id]
            true_canon = canonical_reactant_set(true_raw)
            pred_canon = canonical_reactant_set(top1_smiles)

            if true_canon is None:
                n_unparseable_truth += 1
                continue
            if pred_canon is None:
                n_unparseable_pred += 1
                error = 1  # unparseable prediction counts as wrong
            else:
                error = 0 if pred_canon == true_canon else 1

            n_total += 1
            n_error += error

            rows_out.append({
                'mol_idx': test_id,
                'top_score': top_score,
                'uq': uq,
                'pred_top1': top1_smiles,
                'true_reactants': true_raw,
                'error': error
            })

    print('Processed %d molecules.' % n_total)
    print('Top-1 accuracy: %.4f (%d / %d correct)' % (1 - n_error / n_total, n_total - n_error, n_total))
    print('Unparseable predictions (counted as error): %d' % n_unparseable_pred)
    print('Unparseable ground-truth (skipped): %d' % n_unparseable_truth)

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mol_idx', 'top_score', 'uq', 'pred_top1', 'true_reactants', 'error'])
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    print('Wrote %s' % args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LocalRetro baseline calibration table builder')
    parser.add_argument('--decoded', default='../outputs/decoded_prediction/LocalRetro_USPTO_50K.txt',
                         help='Path to decoded prediction file')
    parser.add_argument('--raw_test', default='../data/USPTO_50K/raw_test.csv',
                         help='Path to raw_test.csv (atom-mapped ground truth)')
    parser.add_argument('--output', default='../outputs/localretro_baseline.csv',
                         help='Output path for the calibration CSV')
    args = parser.parse_args()
    main(args)
