"""
MC Dropout utilities for LocalRetro.

Drop this file into the same `scripts/` directory as utils.py and get_edit.py.
It does NOT modify utils.py or get_edit.py -- it imports from them and adds
new functions on top.
"""

import torch
import torch.nn as nn
import numpy as np
import dgl

from utils import predict
from get_edit import combined_edit, get_bg_partition


def enable_dropout(model):
    """
    Set the WHOLE model to eval() (so batchnorm / other eval-sensitive layers
    behave correctly), then walk every submodule and switch only nn.Dropout
    layers back to train() mode. This is model-agnostic -- it does not need
    to know where dropout lives inside LocalRetro_model.
    """
    model.eval()
    n_dropout_layers = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
            n_dropout_layers += 1
    return n_dropout_layers


def mc_predict_batch(args, model, bg, n_passes):
    """
    Run N stochastic forward passes on a single batched graph.
    Returns lists of length n_passes, each element being the
    (atom_logits, bond_logits) softmax tensors for that pass.

    NOTE: utils.predict() does bg.ndata.pop('h') / bg.edata.pop('e'),
    which destructively removes the features from the graph after the
    first call. Since we need to run the forward pass N times on the
    SAME graph, we extract the features once up front (without popping)
    and call the model directly for every subsequent pass instead of
    going through predict() again.
    """
    bg = bg.to(args['device'])
    node_feats = bg.ndata['h'].to(args['device'])  # use [] not pop() -- keep it on bg for re-use
    edge_feats = bg.edata['e'].to(args['device'])

    atom_probs_list = []
    bond_probs_list = []

    with torch.no_grad():
        for _ in range(n_passes):
            batch_atom_logits, batch_bond_logits, _ = model(bg, node_feats, edge_feats)
            batch_atom_logits = nn.Softmax(dim=1)(batch_atom_logits)
            batch_bond_logits = nn.Softmax(dim=1)(batch_bond_logits)
            atom_probs_list.append(batch_atom_logits)
            bond_probs_list.append(batch_bond_logits)

    return atom_probs_list, bond_probs_list


def write_edits_mc_dropout(args, model, test_loader, n_passes=20):
    """
    MC-dropout version of write_edits (see get_edit.py).

    Writes two files:
      1. args['result_path']        -- same format as baseline Test.py output,
                                        but predictions/scores are AVERAGED
                                        over n_passes stochastic forward passes.
                                        Drop-in compatible with Decode_predictions.py.
      2. args['mc_uq_path']         -- one row per molecule:
                                        mol_idx, mean_top_score, std_top_score,
                                        agreement_rate, predictive_entropy

    NOTE: bg is re-used (same batched graph object) across the n_passes
    forward passes -- this is required so dropout is the ONLY source of
    stochasticity, i.e. we are not re-batching or re-featurizing.
    """
    n_dropout_layers = enable_dropout(model)
    print('MC Dropout enabled on %d nn.Dropout layer(s). Running %d passes per batch.'
          % (n_dropout_layers, n_passes))
    if n_dropout_layers == 0:
        print('WARNING: no nn.Dropout layers found in model.modules(). '
              'MC dropout will produce IDENTICAL passes (zero variance). '
              'Check models.py for how dropout is implemented (e.g. functional '
              'F.dropout calls instead of nn.Dropout submodules).')

    with open(args['result_path'], 'w') as f_pred, open(args['mc_uq_path'], 'w') as f_uq:
        f_pred.write('Test_id\tProduct\t%s\n' % '\t'.join(
            ['Prediction %s' % (i + 1) for i in range(args['top_num'])]))
        f_uq.write('mol_idx\tmean_top_score\tstd_top_score\tagreement_rate\tpredictive_entropy\n')

        for batch_id, data in enumerate(test_loader):
            smiles_list, bg, rxns = data
            bg = bg.to(args['device'])

            # Run N stochastic passes on this batch
            atom_probs_list, bond_probs_list = mc_predict_batch(args, model, bg, n_passes)

            # Partition graph once (topology doesn't change across passes)
            graphs, nodes_sep, edges_sep = get_bg_partition(bg)

            print('\rWriting MC-dropout test molecule batch %s/%s' %
                  (batch_id, len(test_loader)), end='', flush=True)

            start_node = 0
            start_edge = 0
            for single_id, (graph, end_node, end_edge) in enumerate(zip(graphs, nodes_sep, edges_sep)):
                smiles = smiles_list[single_id]
                test_id = (batch_id * args['batch_size']) + single_id

                # --- per-pass top-1 predictions for this molecule ---
                per_pass_top1 = []   # list of (type, site_tuple, score) across passes
                per_pass_mean_atom = torch.zeros_like(atom_probs_list[0][start_node:end_node])
                per_pass_mean_bond = torch.zeros_like(bond_probs_list[0][start_edge:end_edge])

                for p in range(n_passes):
                    atom_slice = atom_probs_list[p][start_node:end_node]
                    bond_slice = bond_probs_list[p][start_edge:end_edge]
                    per_pass_mean_atom += atom_slice
                    per_pass_mean_bond += bond_slice

                    pred_types, pred_sites, pred_scores = combined_edit(
                        graph, atom_slice, bond_slice, top_num=1)
                    per_pass_top1.append((pred_types[0], pred_sites[0], pred_scores[0]))

                per_pass_mean_atom /= n_passes
                per_pass_mean_bond /= n_passes

                # --- averaged prediction (drop-in compatible top-k output) ---
                pred_types, pred_sites, pred_scores = combined_edit(
                    graph, per_pass_mean_atom, per_pass_mean_bond, args['top_num'])

                f_pred.write('%s\t%s\t%s\n' % (
                    test_id, smiles,
                    '\t'.join(['(%s, %s, %s, %.3f)' % (
                        pred_types[i], pred_sites[i][0], pred_sites[i][1], pred_scores[i])
                        for i in range(args['top_num'])])
                ))

                # --- MC dropout uncertainty signals ---
                consensus_type, consensus_site = pred_types[0], pred_sites[0]
                pass_scores = np.array([s for (_, _, s) in per_pass_top1])
                mean_top_score = float(pass_scores.mean())
                std_top_score = float(pass_scores.std())

                agreements = [
                    1 if (t == consensus_type and tuple(site) == tuple(consensus_site)) else 0
                    for (t, site, s) in per_pass_top1
                ]
                agreement_rate = float(np.mean(agreements))

                # predictive entropy of the averaged top-1 site's score
                # (binary entropy: confident in this template vs not)
                eps = 1e-12
                p1 = min(max(mean_top_score, eps), 1 - eps)
                predictive_entropy = float(-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)))

                f_uq.write('%d\t%.6f\t%.6f\t%.6f\t%.6f\n' % (
                    test_id, mean_top_score, std_top_score, agreement_rate, predictive_entropy))

                start_node = end_node
                start_edge = end_edge

    print()
    return
