import gzip
import os.path as pt

import scipy.io as sio


_call_options = []


def _load_mat(location):
    if location.endswith('.gz'):
        return sio.loadmat(
            gzip.open(location, 'rb'),
            squeeze_me=True,
            struct_as_record=False
            )['SPM']

    return sio.loadmat(
        location, squeeze_me=True, struct_as_record=False)['SPM']


def _wdir(wd):
    def func(path):
        return pt.join(str(wd), pt.split(str(path))[1])
    return func


def _find_data_dir(wd, fpath):

    def right_splits(p):
        while p not in ['', None]:
            p = p.rsplit(pt.sep, 1)[0]
            yield p

    def left_splits(p):
        while len(p.split(pt.sep, 1)) > 1:
            p = p.split(pt.sep, 1)[1]
            yield p

    if not pt.isfile(fpath):
        for rs in right_splits(wd):
            for ls in left_splits(fpath):
                p = pt.join(rs, *ls.split(pt.sep))
                if pt.isfile(p):
                    return pt.dirname(p)
    else:
        return pt.dirname(fpath)


def _prefix_filename(path, prefix):
    path, filename = pt.split(str(path))
    return pt.join(path, '%s%s' % (prefix, filename))


def load_intra(spmdotmat_path, inputs=True, outputs=True, **options):
    """Function to load SPM.mat data structure, fixing file paths
    (of images, etc.) as necessary

    Parameters
    ----------

    spmdotmat_path: string
        exisiting filename, path to SPM.mat matrix to be loaded

    Returns
    -------
    A dict of params of the analysis specified in the SPM.mat file

    XXX Document the other args for this function!!!

    """

    spmmat = _load_mat(spmdotmat_path)

    wd, _ = pt.split(pt.realpath(spmdotmat_path))  # work dir
    bd = _wdir(wd)                           # beta directory

    analysis = {}  # this will contain our final stuff to return

    for opt in options:
        if opt not in _call_options:
            if isinstance(options[opt], int):
                analysis[opt] = wd.split(pt.sep)[options[opt]]
            else:
                analysis[opt] = options[opt]

    # parse loaded mat structure for fields we need
    analysis['design_matrix'] = spmmat.xX.X.tolist()  # xX: design
    analysis['conditions'] = [str(i) for i in spmmat.xX.name]  # xX: design
    analysis['n_scans'] = spmmat.nscan.tolist() \
        if isinstance(spmmat.nscan.tolist(), list) else [spmmat.nscan.tolist()]
    analysis['n_sessions'] = spmmat.nscan.size
    analysis['TR'] = float(spmmat.xY.RT)    # xY: data
    analysis['mask'] = bd(spmmat.VM.fname)  # VM: mask

    if outputs:
        analysis['b_maps'] = []
        analysis['c_maps'] = {}
        analysis['c_maps_smoothed'] = {}
        analysis['t_maps'] = {}
        analysis['contrasts'] = {}

        for c in spmmat.xCon:
            name = str(c.name)
            scon = _prefix_filename(c.Vcon.fname, 's')

            analysis['c_maps'][name] = bd(c.Vcon.fname)
            analysis['c_maps_smoothed'][name] = bd(scon)
            analysis['t_maps'][name] = bd(c.Vspm.fname)
            analysis['contrasts'][name] = c.c.tolist()

        for i, b in enumerate(spmmat.Vbeta):
            analysis['b_maps'].append(bd(b.fname))

    if inputs:
        analysis['raw_data'] = []
        analysis['data'] = []
        for Y in spmmat.xY.P:
            Y = str(Y).strip()
            data_dir = _find_data_dir(wd, Y)
            if data_dir is not None:
                analysis['data'].append(pt.join(data_dir, pt.split(Y)[1]))
                analysis['raw_data'].append(
                    pt.join(data_dir, pt.split(Y)[1].strip('swa')))
            else:
                analysis['data'].append(pt.split(Y)[1])
                analysis['raw_data'].append(pt.split(Y)[1].strip('swa'))

    return analysis
