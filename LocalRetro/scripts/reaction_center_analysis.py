"""
reaction_center_analysis.py

For each test molecule, extracts the TRUE reaction center (atoms/bonds whose
local environment changes between atom-mapped reactants and product) and
checks whether the model's top-1 predicted edit site falls within that core.

Cross-tabulates against the existing 'error' column from localretro_baseline.csv:
  - correct prediction + site in core      -> "right answer, right reason"
  - correct prediction + site NOT in core  -> "right answer, wrong reason"  (interesting failure mode)
  - wrong prediction + site in core        -> model found the right region but wrong template
  - wrong prediction + site NOT in core    -> model attended to the wrong place entirely

IMPORTANT: bond-index mapping assumes the standard dgllife smiles_to_bigraph
edge ordering (iterate RDKit bonds, emit forward+reverse directed edge per
bond, self-loops stripped before indexing in get_edit.py). This script prints
a built-in sanity check -- atom/bond site overlap rate AMONG CORRECT
PREDICTIONS should be high (~70%+) if indexing is right. If it's low,
the indexing assumption needs to be revisited before trusting any numbers.

Inputs:
  - ../data/USPTO_50K/raw_test.csv                          (atom-mapped ground truth)
  - ../outputs/raw_prediction/LocalRetro_USPTO_50K.txt      (raw top-1 predicted site)
  - ../outputs/localretro_baseline.csv                      (error labels, for cross-tab)

Output:
  - ../outputs/reaction_center_analysis.csv
  - printed summary stats + cross-tab

Run from scripts/ directory:
    python reaction_center_analysis.py
"""

import argparse
import csv
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


import sys
sys.path.append('../')
from LocalTemplate.template_extractor import (
    split_reagents, clean_map_and_sort, extend_atom_tag,
    get_changed_atoms, set_extractor, default_setting
)

set_extractor(default_setting)


def get_true_reaction_core_and_product_mol(rxn_smiles):
    """
    rxn_smiles: 'reactants>reagents>product' (atom-mapped)

    Uses LocalRetro's OWN get_changed_atoms() function (the exact function
    used to build the templates this checkpoint was trained on) to
    determine which atom-map-numbers are part of the reaction core.
    This mirrors the preprocessing extract_from_reaction() does
    (split_reagents, clean_map_and_sort, extend_atom_tag, RemoveHs/Sanitize)
    before calling get_changed_atoms, so the result is guaranteed
    consistent with the model's actual training labels.

    Returns: (core_mapnums: set of ints, product_mol: RDKit mol parsed
              DIRECTLY from the original atom-mapped product SMILES --
              used only for index lookup, NOT passed through
              clean_map_and_sort, since that step can reorder atoms via
              a canonical-SMILES roundtrip. We only need the MAP NUMBERS
              from get_changed_atoms, not atom indices from the cleaned
              mols, so reordering inside get_changed_atoms is harmless.)

    Returns (None, None) on any parse/preprocessing failure.
    """
    parts = rxn_smiles.split('>')
    if len(parts) != 3:
        return None, None
    reactants_part, reagents_part, products_part = parts

    # original, unmodified parse -- this is what we index INTO later
    original_product_mol = Chem.MolFromSmiles(products_part)
    if original_product_mol is None:
        return None, None

    reaction = {'reactants': reactants_part, 'products': products_part,
                'reagents': reagents_part, '_id': 0}

    try:
        reactants_list, products_list, reagents_list = split_reagents(reaction)
        product_maps = [atom.GetAtomMapNum() for p in products_list
                         for atom in Chem.MolFromSmiles(p).GetAtoms()]
        products = clean_map_and_sort(products_list, product_maps, return_mols=True)
        reactants_ = clean_map_and_sort(reactants_list, product_maps, return_mols=True)
        max_num = max(product_maps)
        reactants = []
        for reactant in reactants_:
            is_reagent, max_num = extend_atom_tag(reactant, max_num)
            if not is_reagent:
                reactants.append(reactant)

        if None in reactants or None in products:
            return None, None

        for i in range(len(reactants)):
            reactants[i] = AllChem.RemoveHs(reactants[i])
        for i in range(len(products)):
            products[i] = AllChem.RemoveHs(products[i])
        for mol in reactants + products:
            Chem.SanitizeMol(mol)
            mol.UpdatePropertyCache()

        changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
        if err:
            return None, None

        core_mapnums = set(int(t) for t in changed_atom_tags)
        return core_mapnums, original_product_mol

    except Exception:
        return None, None


def product_mapnum_to_idx(product_smi):
    """Returns dict: atom_map_number -> RDKit atom index (0-indexed, parse order)."""
    mol = Chem.MolFromSmiles(product_smi)
    if mol is None:
        return None, None
    mapping = {}
    for atom in mol.GetAtoms():
        m = atom.GetAtomMapNum()
        if m:
            mapping[m] = atom.GetIdx()
    return mapping, mol


def build_bond_idx_to_atompair(mol):
    """
    Reconstructs the (atom_i, atom_j) pair for each non-self-loop DGL bond
    index, assuming dgllife's standard smiles_to_bigraph ordering: iterate
    RDKit bonds in order, emit (begin,end) then (end,begin) as consecutive
    directed edges. After self-loop stripping (done upstream in get_edit.py),
    edge index k corresponds to RDKit bond index k // 2, direction k % 2.
    We only need the UNORDERED atom pair for our purposes, so direction
    doesn't matter -- both edges for a given bond map to the same pair.
    """
    bond_idx_to_pair = {}
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bond_idx_to_pair[2 * i] = (a1, a2)
        bond_idx_to_pair[2 * i + 1] = (a1, a2)
    return bond_idx_to_pair


def parse_raw_prediction_top1(line):
    """
    Parses one line of the RAW prediction file (Test.py output, not decoded):
        test_id \t product_smiles \t (type, idx1, idx2, score) \t ...
    Returns (test_id, product_smiles, edit_type, site_idx)
    edit_type: 'a' or 'b'
    For 'a': site is (atom_idx, template_id) -- we want atom_idx
    For 'b': site is (bond_idx, template_id) -- we want bond_idx
    """
    parts = line.rstrip('\n').split('\t')
    test_id = int(parts[0])
    product_smiles = parts[1]
    top1_str = parts[2].strip()
    m = re.match(r'\(([ab]), (\d+), (\d+), ([\d.]+)\)', top1_str)
    if not m:
        return test_id, product_smiles, None, None, None
    edit_type = m.group(1)
    site_idx = int(m.group(2))
    score = float(m.group(4))
    return test_id, product_smiles, edit_type, site_idx, score


def load_raw_test_rxns(path):
    rxns = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rxn_col_idx = None
        for i, col in enumerate(header):
            if 'reactants' in col and 'production' in col:
                rxn_col_idx = i
                break
        for mol_idx, row in enumerate(reader):
            rxns[mol_idx] = row[rxn_col_idx]
    return rxns


def load_errors(path):
    errors = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            errors[int(row['mol_idx'])] = int(row['error'])
    return errors


def main(args):
    print('Loading atom-mapped test reactions...')
    rxns = load_raw_test_rxns(args.raw_test)
    print('Loaded %d reactions.' % len(rxns))

    print('Loading baseline error labels...')
    errors = load_errors(args.baseline)

    print('Processing raw predictions...')
    results = []
    n_skipped_parse = 0
    n_debug_printed = 0
    n_debug2_printed = 0

    with open(args.raw_prediction, 'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line == '':
                continue
            test_id, product_smiles, edit_type, site_idx, score = parse_raw_prediction_top1(line)
            if test_id not in rxns:
                continue
            if edit_type is None:
                n_skipped_parse += 1
                continue

            rxn_smiles = rxns[test_id]
            core_mapnums, mol = get_true_reaction_core_and_product_mol(rxn_smiles)
            if core_mapnums is None:
                n_skipped_parse += 1
                continue

            if args.debug and n_debug_printed < 5:
                print('  [debug-core] mol_idx=%d  core_mapnums=%s' % (test_id, sorted(core_mapnums)))
                n_debug_printed += 1

            core_idx_set = set(
                atom.GetIdx() for atom in mol.GetAtoms()
                if atom.GetAtomMapNum() in core_mapnums
            )

            if edit_type == 'a':
                site_in_core = site_idx in core_idx_set
                pred_mapnums = [mol.GetAtoms()[site_idx].GetAtomMapNum()] if site_idx < mol.GetNumAtoms() else []
            else:  # 'b'
                bond_map = build_bond_idx_to_atompair(mol)
                if site_idx not in bond_map:
                    n_skipped_parse += 1
                    continue
                a1, a2 = bond_map[site_idx]
                site_in_core = (a1 in core_idx_set) or (a2 in core_idx_set)
                pred_mapnums = [mol.GetAtoms()[a1].GetAtomMapNum(), mol.GetAtoms()[a2].GetAtomMapNum()]

            error = errors.get(test_id, None)

            if args.debug and error == 0 and n_debug2_printed < 10:
                print('  [debug-match] mol_idx=%d type=%s  predicted_site_mapnums=%s  true_core_mapnums=%s  overlap=%s' % (
                    test_id, edit_type, pred_mapnums, sorted(core_mapnums), site_in_core))
                n_debug2_printed += 1
            results.append({
                'mol_idx': test_id,
                'edit_type': edit_type,
                'site_idx': site_idx,
                'site_in_core': int(site_in_core),
                'error': error
            })

    print('Processed %d molecules (%d skipped due to parsing issues).' % (len(results), n_skipped_parse))

    # write output
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mol_idx', 'edit_type', 'site_idx', 'site_in_core', 'error'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print('Wrote %s' % args.output)

    # summary stats
    valid = [r for r in results if r['error'] is not None]
    overall_overlap = sum(r['site_in_core'] for r in valid) / len(valid)
    print()
    print('=== Overall site-in-core rate: %.4f (%d / %d) ===' % (overall_overlap, sum(r['site_in_core'] for r in valid), len(valid)))

    correct = [r for r in valid if r['error'] == 0]
    wrong = [r for r in valid if r['error'] == 1]

    if correct:
        c_overlap = sum(r['site_in_core'] for r in correct) / len(correct)
        print('SANITY CHECK -- site-in-core rate AMONG CORRECT predictions: %.4f (%d / %d)' % (
            c_overlap, sum(r['site_in_core'] for r in correct), len(correct)))
        print('  (this should be HIGH, e.g. >0.7, if bond/atom indexing is correct -- '
              'a correct top-1 prediction essentially has to hit the true reaction center)')

        correct_a = [r for r in correct if r['edit_type'] == 'a']
        correct_b = [r for r in correct if r['edit_type'] == 'b']
        if correct_a:
            print('    -> atom-type, correct only: %.4f (%d / %d)' % (
                sum(r['site_in_core'] for r in correct_a) / len(correct_a), sum(r['site_in_core'] for r in correct_a), len(correct_a)))
        if correct_b:
            print('    -> bond-type, correct only: %.4f (%d / %d)' % (
                sum(r['site_in_core'] for r in correct_b) / len(correct_b), sum(r['site_in_core'] for r in correct_b), len(correct_b)))
    if wrong:
        w_overlap = sum(r['site_in_core'] for r in wrong) / len(wrong)
        print('site-in-core rate among INCORRECT predictions: %.4f (%d / %d)' % (
            w_overlap, sum(r['site_in_core'] for r in wrong), len(wrong)))

    # the genuinely interesting cross-tab
    right_right = sum(1 for r in correct if r['site_in_core'] == 1)
    right_wrong = sum(1 for r in correct if r['site_in_core'] == 0)
    print()
    print('=== Cross-tab among CORRECT predictions (error=0) ===')
    print('  Right answer, right reason (site in core):     %d (%.2f%%)' % (right_right, 100 * right_right / max(len(correct), 1)))
    print('  Right answer, WRONG reason (site NOT in core):  %d (%.2f%%)' % (right_wrong, 100 * right_wrong / max(len(correct), 1)))

    # breakdown by edit type
    a_results = [r for r in valid if r['edit_type'] == 'a']
    b_results = [r for r in valid if r['edit_type'] == 'b']
    print()
    print('=== Breakdown by prediction type ===')
    if a_results:
        print('atom-type predictions: %d total, site-in-core rate %.4f' % (
            len(a_results), sum(r['site_in_core'] for r in a_results) / len(a_results)))
    if b_results:
        print('bond-type predictions: %d total, site-in-core rate %.4f' % (
            len(b_results), sum(r['site_in_core'] for r in b_results) / len(b_results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Reaction center interpretability analysis')
    parser.add_argument('--raw_test', default='../data/USPTO_50K/raw_test.csv')
    parser.add_argument('--raw_prediction', default='../outputs/raw_prediction/LocalRetro_USPTO_50K.txt')
    parser.add_argument('--baseline', default='../outputs/localretro_baseline.csv')
    parser.add_argument('--output', default='../outputs/reaction_center_analysis.csv')
    parser.add_argument('--debug', action='store_true', help='Print core_mapnums for first 10 molecules for manual inspection')
    args = parser.parse_args()
    main(args)
