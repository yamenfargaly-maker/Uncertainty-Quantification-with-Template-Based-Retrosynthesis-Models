# first line: 504
def smarts2appl(product_smarts, template_product_smarts, fpsize=2048, v=False, use_tqdm=False, njobs=1, nsplits=1):
    """This takes in a list of product smiles (misnamed in code) and a list of product sides
    of templates and calculates which templates are applicable to which product.
    This is basically a substructure search. Maybe there are faster versions but I wrote this one.

    Args:
        product_smarts: List of smiles of molecules to check.
        template_product_smarts: List of substructures to check
        fpsize: fingerprint size to use in screening
        v: if v then information will be printed
        use_tdqm: if True then a progressbar will be displayed but slows down the computation.
        njobs: how many parallel jobs to run in parallel.
        nsplits: how many splits should be made along the product_smarts list. Useful to avoid memory
            explosion.
    Returns: list of tuples (i,j) that indicates the product i has substructure j.
    """
    if v: print("Calculating template molecules")
    template_mols = [Chem.MolFromSmarts(s) for s in template_product_smarts]
    if v: print("Calculating template fingerprints")
    template_ebvs = [Chem.PatternFingerprint(m, fpSize=fpsize) for m in template_mols]
    if v: print(f'Building template ints: [{len(template_mols)}, {fpsize}]')
    template_ints = [int(e.ToBitString(), base=2) for e in template_ebvs]
    del template_ebvs

    if njobs == 1 and nsplits == 1:
        return _smarts2appl(product_smarts, template_product_smarts, template_ints, fpsize, v, use_tqdm)
    elif nsplits == 1:
        nsplits = njobs


    # split products into batches
    product_splits = np.array_split(np.array(product_smarts), nsplits)
    ioffsets = [0] + list(np.cumsum([p.shape[0] for p in product_splits[:-1]]))
    inps = [(ps, template_product_smarts, template_ints, fpsize, v, use_tqdm, ioff, 0) for ps, ioff in zip(product_splits, ioffsets)]

    if v: print("Creating workers")
    #results = process_map(__smarts2appl, inps, max_workers=njobs, chunksize=1)
    with Pool(njobs) as pool:
        results = pool.starmap(_smarts2appl, inps)
    imatch = np.concatenate([r[0] for r in results])
    jmatch = np.concatenate([r[1] for r in results])
    return imatch, jmatch
