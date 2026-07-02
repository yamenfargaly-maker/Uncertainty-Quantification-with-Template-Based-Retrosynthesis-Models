"""
tta_utils.py + Test_tta.py logic combined.

Test-Time Augmentation (TTA) via random SMILES reordering.

Unlike MC dropout (which perturbs the MODEL via stochastic dropout while
keeping the input fixed), this perturbs the INPUT: for each test molecule,
we generate N randomly-reordered (but chemically identical) SMILES strings
of the product, rebuild the DGL graph from each, and run a normal
(deterministic, dropout OFF) forward pass each time.

This tests whether the model's prediction and confidence are invariant to
an arbitrary, chemically meaningless choice (atom ordering in the input
string) -- a property a well-behaved GNN should have, but is not
guaranteed by architecture alone (e.g. attention layers, batch norm
statistics, or numerical floating-point order-of-operations can introduce
small dependencies on input ordering).

Because each augmented pass uses a DIFFERENT atom ordering, raw atom/bond
INDICES are not directly comparable across passes. To track whether the
"same" edit was predicted each time, we tag each atom with its ORIGINAL
index via a temporary atom-map number before randomizing, then map
predictions back to that stable identity after each pass.

Run from scripts/ directory:
    python Test_tta.py -d USPTO_50K -n 20
"""

import argparse
import csv

import torch
import torch.nn as nn
import numpy as np
import dgl
from functools import partial

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dgllife.utils import smiles_to_bigraph

from utils import init_featurizer, mkdir_p, get_configure, load_model
from get_edit import combined_edit, get_bg_partition


def randomize_smiles_with_tracking(mol):
    """
    Returns a randomly-reordered RDKit mol of the same molecule, where each
    atom's AtomMapNum is set to its ORIGINAL (pre-randomization) atom index
    + 1 (RDKit map numbers must be >0), so we can recover the original
    identity of any atom after reordering.
    """
    mol = Chem.Mol(mol)  # copy
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    new_mol = Chem.MolFromSmiles(random_smiles)
    return new_mol  # atoms carry AtomMapNum = original_idx + 1


def original_idx(mol, new_idx):
    return mol.GetAtomWithIdx(int(new_idx)).GetAtomMapNum() - 1


def build_bond_idx_to_origpair(mol):
    """Same convention as reaction_center_analysis.py: edge k -> RDKit bond k//2, both directions map to same pair."""
    bond_idx_to_pair = {}
    for i, bond in enumerate(mol.GetBonds()):
        a1 = original_idx(mol, bond.GetBeginAtomIdx())
        a2 = original_idx(mol, bond.GetEndAtomIdx())
        bond_idx_to_pair[2 * i] = (a1, a2)
        bond_idx_to_pair[2 * i + 1] = (a1, a2)
    return bond_idx_to_pair


def run_single_molecule_tta(args, model, product_smiles_clean, n_passes):
    """
    product_smiles_clean: the unmapped product SMILES as used at normal inference.
    Returns dict of TTA summary stats for this molecule, or None on failure.
    """
    base_mol = Chem.MolFromSmiles(product_smiles_clean)
    if base_mol is None:
        return None

    per_pass_records = []  # (edit_type, frozenset(orig_atom_ids), template_id, score)

    model.eval()  # dropout OFF -- pure input augmentation, no stochastic model behavior
    with torch.no_grad():
        for _ in range(n_passes):
            aug_mol = randomize_smiles_with_tracking(base_mol)
            if aug_mol is None:
                continue

            g = smiles_to_bigraph(
                Chem.MolToSmiles(aug_mol), add_self_loop=True,
                node_featurizer=args['node_featurizer'],
                edge_featurizer=args['edge_featurizer']
            )
            # re-parse to get a mol object whose atom order matches the graph exactly
            graph_mol = Chem.MolFromSmiles(Chem.MolToSmiles(aug_mol))
            # NOTE: re-parsing via MolToSmiles->MolFromSmiles roundtrip again could
            # in principle reorder atoms again. To avoid that risk entirely, we
            # build the graph from aug_mol's own canonical-from-itself smiles AND
            # immediately re-derive graph_mol from THAT exact string, so graph_mol
            # and g are guaranteed to agree on atom order (both come from the same
            # string -> same RDKit parse).

            bg = dgl.batch([g])
            bg.set_n_initializer(dgl.init.zero_initializer)
            bg.set_e_initializer(dgl.init.zero_initializer)
            bg = bg.to(args['device'])
            node_feats = bg.ndata['h'].to(args['device'])
            edge_feats = bg.edata['e'].to(args['device'])

            atom_logits, bond_logits, _ = model(bg, node_feats, edge_feats)
            atom_probs = nn.Softmax(dim=1)(atom_logits)
            bond_probs = nn.Softmax(dim=1)(bond_logits)

            sg = bg.remove_self_loop()
            graphs = dgl.unbatch(sg)
            graph = graphs[0]

            pred_types, pred_sites, pred_scores = combined_edit(graph, atom_probs, bond_probs, top_num=1)
            edit_type = pred_types[0]
            site = pred_sites[0]
            score = pred_scores[0]

            if edit_type == 'a':
                orig_ids = frozenset([original_idx(graph_mol, site[0])])
            else:
                bond_map = build_bond_idx_to_origpair(graph_mol)
                if site[0] not in bond_map:
                    continue
                a1, a2 = bond_map[site[0]]
                orig_ids = frozenset([a1, a2])

            template_id = site[1]
            per_pass_records.append((edit_type, orig_ids, template_id, score))

    if len(per_pass_records) == 0:
        return None

    # consensus = most common (edit_type, orig_ids, template_id) combination across passes
    from collections import Counter
    key_counts = Counter((t, fs, tpl) for (t, fs, tpl, s) in per_pass_records)
    consensus_key, consensus_count = key_counts.most_common(1)[0]

    scores = np.array([s for (t, fs, tpl, s) in per_pass_records])
    agreement_rate = consensus_count / len(per_pass_records)
    mean_top_score = float(scores.mean())
    std_top_score = float(scores.std())

    return {
        'n_valid_passes': len(per_pass_records),
        'mean_top_score': mean_top_score,
        'std_top_score': std_top_score,
        'agreement_rate': agreement_rate,
    }


def load_raw_test_products(path):
    """Returns dict mol_idx -> unmapped product SMILES (matches what Test.py's 'Product' column shows)."""
    products = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rxn_col_idx = None
        for i, col in enumerate(header):
            if 'reactants' in col and 'production' in col:
                rxn_col_idx = i
                break
        for mol_idx, row in enumerate(reader):
            rxn = row[rxn_col_idx]
            product_mapped = rxn.split('>')[-1]
            mol = Chem.MolFromSmiles(product_mapped)
            if mol is None:
                products[mol_idx] = None
                continue
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            products[mol_idx] = Chem.MolToSmiles(mol)
    return products


def main(args):
    model_name = 'LocalRetro_%s.pth' % args['dataset']
    args['model_path'] = '../models/%s' % model_name
    args['config_path'] = '../data/configs/%s' % args['config']
    args['data_dir'] = '../data/%s' % args['dataset']
    args['raw_test_path'] = '../data/%s/raw_test.csv' % args['dataset']
    args['output_path'] = '../outputs/raw_prediction/%s' % model_name.replace('.pth', '_tta_uq.txt')
    mkdir_p('../outputs')
    mkdir_p('../outputs/raw_prediction')

    args = init_featurizer(args)
    model = load_model(args)

    print('Loading product SMILES from %s ...' % args['raw_test_path'])
    products = load_raw_test_products(args['raw_test_path'])
    print('Loaded %d products.' % len(products))

    with open(args['output_path'], 'w') as f:
        f.write('mol_idx\tmean_top_score\tstd_top_score\tagreement_rate\tn_valid_passes\n')
        for mol_idx, smi in products.items():
            if smi is None:
                continue
            result = run_single_molecule_tta(args, model, smi, args['n_passes'])
            if result is None:
                continue
            f.write('%d\t%.6f\t%.6f\t%.6f\t%d\n' % (
                mol_idx, result['mean_top_score'], result['std_top_score'],
                result['agreement_rate'], result['n_valid_passes']))
            if mol_idx % 50 == 0:
                print('\rTTA progress: molecule %d / %d' % (mol_idx, len(products)), end='', flush=True)

    print()
    print('Wrote %s' % args['output_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LocalRetro attention/input TTA')
    parser.add_argument('-g', '--gpu', default='cuda:0')
    parser.add_argument('-d', '--dataset', default='USPTO_50K')
    parser.add_argument('-c', '--config', default='default_config.json')
    parser.add_argument('-n', '--n_passes', default=20, type=int, help='Number of randomized-SMILES augmentation passes per molecule')
    args = parser.parse_args().__dict__
    args['mode'] = 'test'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    main(args)
