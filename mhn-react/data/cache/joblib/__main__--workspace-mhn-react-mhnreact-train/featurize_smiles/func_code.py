# first line: 135
@memory.cache(ignore=['njobs'])
def featurize_smiles(X, fp_type='morgan', fp_size=4096, fp_radius=2, njobs=1, verbose=False):
    X_fp = {}
    
    if fp_type in ['MxFP','MACCS','Morgan2CBF','Morgan4CBF', 'Morgan6CBF', 'ErG','AtomPair','TopologicalTorsion','RDK']:
        print('computing', fp_type)
        if fp_type == 'MxFP':
            fp_types = ['MACCS','Morgan2CBF','Morgan4CBF', 'Morgan6CBF', 'ErG','AtomPair','TopologicalTorsion','RDK']
        else:
            fp_types = [fp_type]

        remaining = int(fp_size)
        for fp_type in fp_types:
            print(fp_type,end=' ')
            feat = FP_featurizer(fp_types=fp_type,
                                 max_features= (fp_size//len(fp_types)) if (fp_type != fp_types[-1]) else remaining )
            X_fp[f'train_{fp_type}'] = feat.fit(X['train'])
            X_fp[f'valid_{fp_type}'] = feat.transform(X['valid'])
            X_fp[f'test_{fp_type}'] = feat.transform(X['test'])

            remaining -= X_fp[f'train_{fp_type}'].shape[1]
            #X_fp['train'].shape, X_fp['test'].shape
        X_fp['train'] = np.hstack([ X_fp[f'train_{fp_type}'] for fp_type in fp_types])
        X_fp['valid'] = np.hstack([ X_fp[f'valid_{fp_type}'] for fp_type in fp_types])
        X_fp['test'] = np.hstack([ X_fp[f'test_{fp_type}'] for fp_type in fp_types])
    
    else: #fp_type in ['rdk','morgan','ecfp4','pattern','morganc','rdkc']:
        if verbose: print('computing', fp_type, 'folded')
        for split in X.keys():
            X_fp[split] = convert_smiles_to_fp(X[split], fp_size=fp_size, which=fp_type, radius=fp_radius, njobs=njobs, verbose=verbose)

    return X_fp
