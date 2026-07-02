"""
attention_entropy_depth.py

Corrected re-implementation of "attention TTA" (depth-varying attention
entropy as a UQ signal).

FIXES vs. the originally-uploaded localretro_attention_tta.py:
  1. Site selection: the original script always probed attention at the
     atom with the highest ATOM-template score, ignoring bond-type
     predictions entirely. Since ~77% of correct LocalRetro predictions
     are bond-type (per reaction_center_analysis_v2.py), this means the
     wrong attention row was being read for most molecules. This version
     reads the REAL top-1 (type, site_idx) from the raw prediction file
     -- the same ground truth used throughout this entire pipeline.
  2. Bond-site indexing: model_utils.unbatch_mask() concatenates ATOMS
     then BONDS into a single sequence per molecule before computing
     attention (torch.cat((atom_feats, bond_feats), dim=0)). For a
     bond-type prediction at bond index i, the correct attention sequence
     position is (n_atoms + i), NOT i. The original script never handled
     this distinction.
  3. Processes one molecule at a time (batch size 1) to avoid any risk of
     batched-padding offset errors, since unbatch_mask pads each
     molecule's (n_atoms + n_bonds)-length sequence independently before
     stacking -- per-molecule valid entries always start at index 0 of
     that molecule's own row regardless of batch padding, which is only
     safe to rely on with explicit single-molecule processing.

Method: for each molecule, run the model at MPNN depths 1-6 (the model
was trained at depth 6; depths 1-5 evaluate it outside its trained
operating point -- see caveat in report). At each depth, extract the
attention row at the model's predicted top-1 edit site (atom or bond,
correctly resolved per above), averaged across heads, restricted to
valid (non-padding) positions, and compute Shannon entropy. Aggregate
mean/std/range of entropy across the 6 depths per molecule.

Inputs:
  - ../models/LocalRetro_USPTO_50K.pth
  - ../data/USPTO_50K/raw_test.csv
  - ../outputs/raw_prediction/LocalRetro_USPTO_50K.txt   (for real top-1 site)

Output:
  - ../outputs/attention_entropy_depth.csv
    columns: mol_idx, edit_type, site_idx, entropy_mean, entropy_std, entropy_range

Run from scripts/ directory:
    python attention_entropy_depth.py -n_passes_depths 1,2,3,4,5,6
"""

import argparse
import csv
import re
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import dgl

from utils import init_featurizer, mkdir_p, load_model
from dgllife.utils import smiles_to_bigraph
from model_utils import unbatch_mask

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def parse_raw_prediction_top1(line):
    parts = line.rstrip('\n').split('\t')
    test_id = int(parts[0])
    product_smiles = parts[1]
    top1_str = parts[2].strip()
    m = re.match(r'\(([ab]), (\d+), (\d+), ([\d.]+)\)', top1_str)
    if not m:
        return test_id, product_smiles, None, None, None
    edit_type = m.group(1)
    site_idx = int(m.group(2))
    template_id = int(m.group(3))
    score = float(m.group(4))
    return test_id, product_smiles, edit_type, site_idx, score


def load_top1_sites(path):
    sites = {}
    with open(path, 'r') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line == '':
                continue
            test_id, product_smiles, edit_type, site_idx, score = parse_raw_prediction_top1(line)
            if edit_type is None:
                continue
            sites[test_id] = (product_smiles, edit_type, site_idx)
    return sites


def shannon_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def run_single_molecule(model, args, product_smiles, edit_type, site_idx, depths):
    """
    Returns dict of {depth: entropy} for this molecule, or None on failure.
    """
    g = smiles_to_bigraph(product_smiles, add_self_loop=True,
                           node_featurizer=args['node_featurizer'],
                           edge_featurizer=args['edge_featurizer'])
    bg = dgl.batch([g])
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    bg = bg.to(args['device'])

    node_feats = bg.ndata['h'].to(args['device'])
    edge_feats = bg.edata['e'].to(args['device'])

    n_atoms = g.num_nodes()  # includes self-loops? -- smiles_to_bigraph with add_self_loop=True
    # NOTE: self-loops are stripped by unbatch_mask internally via bg.remove_self_loop(),
    # so n_atoms here must reflect the SELF-LOOP-REMOVED graph's atom count, which equals
    # the true atom count of the molecule (self-loops don't add new nodes, only edges).
    n_atoms_real = g.num_nodes()

    if edit_type == 'a':
        if site_idx >= n_atoms_real:
            return None
        seq_pos = site_idx
    else:  # 'b' -- bonds come AFTER all atoms in the concatenated sequence
        seq_pos = n_atoms_real + site_idx

    entropies = {}
    original_depth = model.mpnn.num_step_message_passing

    with torch.no_grad():
        for depth in depths:
            model.mpnn.num_step_message_passing = depth
            atom_out, bond_out, attention_score = model(bg, node_feats, edge_feats)
            # attention_score is a list (one per attention layer); use the last layer
            scores = attention_score[-1]  # shape [batch=1, heads, seq_len, seq_len]

            seq_len = scores.shape[-1]
            if seq_pos >= seq_len:
                continue

            # average over heads -> [seq_len, seq_len], take row seq_pos
            attn_row = scores[0].mean(dim=0)[seq_pos, :]  # [seq_len]

            # batch size is 1 here, so there is no cross-molecule padding --
            # seq_len already equals this molecule's exact (n_atoms + n_bonds) length.
            probs = attn_row.cpu().numpy()
            probs = probs / (probs.sum() + 1e-12)
            entropies[depth] = shannon_entropy(probs)

    model.mpnn.num_step_message_passing = original_depth

    if len(entropies) == 0:
        return None
    return entropies


def main(args):
    model_name = 'LocalRetro_%s.pth' % args['dataset']
    args['model_path'] = '../models/%s' % model_name
    args['config_path'] = '../data/configs/%s' % args['config']
    args['data_dir'] = '../data/%s' % args['dataset']
    args['raw_prediction_path'] = '../outputs/raw_prediction/%s' % model_name.replace('.pth', '.txt')
    args['output_path'] = '../outputs/attention_entropy_depth.csv'
    mkdir_p('../outputs')

    args = init_featurizer(args)
    model = load_model(args)
    model.eval()

    print('Loading top-1 prediction sites from %s ...' % args['raw_prediction_path'])
    sites = load_top1_sites(args['raw_prediction_path'])
    print('Loaded %d molecules.' % len(sites))

    depths = [int(d) for d in args['depths'].split(',')]
    print('Running depth-varying attention entropy at depths %s ...' % depths)

    results = []
    n_failed = 0
    for i, (mol_idx, (product_smiles, edit_type, site_idx)) in enumerate(sites.items()):
        entropies_dict = run_single_molecule(model, args, product_smiles, edit_type, site_idx, depths)
        if entropies_dict is None:
            n_failed += 1
            continue
        ents = list(entropies_dict.values())
        results.append({
            'mol_idx': mol_idx,
            'edit_type': edit_type,
            'site_idx': site_idx,
            'n_depths_used': len(ents),
            'entropy_mean': float(np.mean(ents)),
            'entropy_std': float(np.std(ents)),
            'entropy_range': float(max(ents) - min(ents)),
        })
        if i % 200 == 0:
            print('\rProgress: %d / %d' % (i, len(sites)), end='', flush=True)

    print()
    print('Processed %d molecules (%d failed/skipped).' % (len(results), n_failed))

    with open(args['output_path'], 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'mol_idx', 'edit_type', 'site_idx', 'n_depths_used',
            'entropy_mean', 'entropy_std', 'entropy_range'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print('Wrote %s' % args['output_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Corrected depth-varying attention entropy UQ')
    parser.add_argument('-g', '--gpu', default='cuda:0')
    parser.add_argument('-d', '--dataset', default='USPTO_50K')
    parser.add_argument('-c', '--config', default='default_config.json')
    parser.add_argument('--depths', default='1,2,3,4,5,6', help='Comma-separated MPNN depths to evaluate')
    args = parser.parse_args().__dict__
    args['mode'] = 'test'
    args['device'] = torch.device(args['gpu']) if torch.cuda.is_available() else torch.device('cpu')
    print('Using device %s' % args['device'])
    main(args)
