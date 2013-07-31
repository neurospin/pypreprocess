"""
:Author: dohmatob elvis dopgima <gmdopp@gmail.com>

"""

cimport cython
cimport numpy as np
import numpy as np

cdef extern from "spm_hist2.h":
    float samp(int *, unsigned char *, float, float, float)


cdef extern from "spm_hist2.h":
    void hist2(double *, unsigned char *, unsigned char *, int *, int *, 
               double *, float *)


def _samppy(np.ndarray[np.int32_t, ndim=1] d,
         np.ndarray[np.uint8_t, ndim=1] f,
         float x,
         float y,
         float z
         ):

    return samp(<int *>d.data, <unsigned char *>f.data, x, y, z)


def _hist2py(np.ndarray[np.double_t, ndim=1] M,
            np.ndarray[np.uint8_t, ndim=1] g,
            np.ndarray[np.uint8_t, ndim=1] f,
            np.ndarray[np.int32_t, ndim=1] dg,
            np.ndarray[np.int32_t, ndim=1] df,
            np.ndarray[np.float32_t, ndim=1] s):

    cdef np.ndarray H = np.zeros(256 * 256, dtype=np.double)

    hist2(<double *>M.data, <unsigned char *>g.data, <unsigned char *>f.data,
           <int *>dg.data, <int *>df.data, <double *>H.data, <float *>s.data)

    return H.reshape((256, 256), order='F')


def samppy(f, x, y, z):
    f = np.array(f, dtype=np.uint8)

    assert f.ndim == 3

    df = np.array(f.shape, dtype=np.int32)
    
    return _samppy(df, f.ravel(order='F'), x, y, z)

def hist2py(M, g, f, s):
    M = np.array(M, dtype=np.double)
    g = np.array(g, dtype=np.uint8)
    f = np.array(f, dtype=np.uint8)
    s = np.array(s, dtype=np.float32)
    
    assert M.shape == (4, 4)
    assert g.ndim == f.ndim == 3
    assert s.ndim == 1
    assert len(s.shape) == 1

    dg = np.array(g.shape, dtype=np.int32)
    df = np.array(f.shape, dtype=np.int32)

    return _hist2py(M.ravel(order='F'), g.ravel(order='F'), f.ravel(order='F'), dg, df, s)