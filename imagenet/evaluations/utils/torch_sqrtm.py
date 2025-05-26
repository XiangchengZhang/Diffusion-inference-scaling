import torch 
import numpy as np

from scipy._lib._util import _asarray_validated

# Local imports
from scipy.linalg._misc import norm
from scipy.linalg.lapack import ztrsyl, dtrsyl
from scipy.linalg._decomp_schur import schur, rsf2csf


class SqrtmError(np.linalg.LinAlgError):
    pass

from scipy.linalg._matfuncs_sqrtm_triu import within_block_loop

def np_to_gpu_tensor(device, np_array):
    return torch.from_numpy(np_array).to(device)

def torch_matmul_to_array(tensor1, tensor2):
    return (tensor1 @ tensor2).cpu().numpy()

def sqrtm(A, array_to_tensor, disp=True, blocksize=64):

    byte_size = np.asarray(A).dtype.itemsize
    A = _asarray_validated(A, check_finite=True, as_inexact=True)
    if len(A.shape) != 2:
        raise ValueError("Non-matrix input to matrix function.")
    if blocksize < 1:
        raise ValueError("The blocksize should be at least 1.")
    keep_it_real = np.isrealobj(A)
    if keep_it_real:
        T, Z = schur(A)
        if not np.array_equal(T, np.triu(T)):
            T, Z = rsf2csf(T, Z)
    else:
        T, Z = schur(A, output='complex')
    failflag = False
    try:
        R = array_to_tensor(_sqrtm_triu(T, array_to_tensor, blocksize=blocksize))
        ZH = array_to_tensor(np.conjugate(Z).T)
        Z = array_to_tensor(Z)

        if torch.is_complex(Z) and not torch.is_complex(R):
            R = torch.complex(R, torch.zeros_like(R))
        elif torch.is_complex(R) and not torch.is_complex(Z):
            Z = torch.complex(Z, torch.zeros_like(Z))       

        Z_mul_R = Z @ R

        if torch.is_complex(Z_mul_R) and not torch.is_complex(ZH):
            ZH = torch.complex(ZH, torch.zeros_like(ZH))
        elif torch.is_complex(ZH) and not torch.is_complex(Z_mul_R):
            Z_mul_R = torch.complex(Z_mul_R, torch.zeros_like(Z_mul_R))
        
        X = torch_matmul_to_array(Z_mul_R, ZH)
        if not np.iscomplexobj(X):
            # float byte size range: f2 ~ f16
            X = X.astype(f"f{np.clip(byte_size, 2, 16)}", copy=False)
        else:
            # complex byte size range: c8 ~ c32.
            # c32(compl ex256) might not be supported in some environments.
            if hasattr(np, 'complex256'):
                X = X.astype(f"c{np.clip(byte_size*2, 8, 32)}", copy=False)
            else:
                X = X.astype(f"c{np.clip(byte_size*2, 8, 16)}", copy=False)
    except SqrtmError:
        failflag = True
        X = np.empty_like(A)
        X.fill(np.nan)

    if disp:
        if failflag:
            print("Failed to find a square root.")
        return X
    else:
        try:
            X_ = array_to_tensor(X)
            X_dot_X = torch_matmul_to_array(X_, X_)
            arg2 = norm(X_dot_X - A, 'fro')**2 / norm(A, 'fro')
        except ValueError:
            # NaNs in matrix
            arg2 = np.inf

        return X, arg2

def _sqrtm_triu(T, array_to_tensor, blocksize=64):

    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag) >= 0

    # Cast to complex as necessary + ensure double precision
    if not keep_it_real:
        T = np.asarray(T, dtype=np.complex128, order="C")
        T_diag = np.asarray(T_diag, dtype=np.complex128)
    else:
        T = np.asarray(T, dtype=np.float64, order="C")
        T_diag = np.asarray(T_diag, dtype=np.float64)

    R = np.diag(np.sqrt(T_diag))

    # Compute the number of blocks to use; use at least one block.
    n, n = T.shape
    nblocks = max(n // blocksize, 1)

    # Compute the smaller of the two sizes of blocks that
    # we will actually use, and compute the number of large blocks.
    bsmall, nlarge = divmod(n, nblocks)
    blarge = bsmall + 1
    nsmall = nblocks - nlarge
    if nsmall * bsmall + nlarge * blarge != n:
        raise Exception('internal inconsistency')

    # Define the index range covered by each block.
    start_stop_pairs = []
    start = 0
    for count, size in ((nsmall, bsmall), (nlarge, blarge)):
        for i in range(count):
            start_stop_pairs.append((start, start + size))
            start += size

    # Within-block interactions (Cythonized)
    try:
        within_block_loop(R, T, start_stop_pairs, nblocks)
    except RuntimeError as e:
        raise SqrtmError(*e.args) from e

    # Between-block interactions (Cython would give no significant speedup)
    for j in range(nblocks):
        jstart, jstop = start_stop_pairs[j]
        for i in range(j-1, -1, -1):
            istart, istop = start_stop_pairs[i]
            S = T[istart:istop, jstart:jstop]
            if j - i > 1:
                R_1 = array_to_tensor(R[istart:istop, istop:jstart])
                R_2 = array_to_tensor(R[istop:jstart, jstart:jstop])
                S = S - torch_matmul_to_array(R_1, R_2)

            # Invoke LAPACK.
            # For more details, see the solve_sylvester implemention
            # and the fortran dtrsyl and ztrsyl docs.
            Rii = R[istart:istop, istart:istop]
            Rjj = R[jstart:jstop, jstart:jstop]
            if keep_it_real:
                x, scale, info = dtrsyl(Rii, Rjj, S)
            else:
                x, scale, info = ztrsyl(Rii, Rjj, S)
            R[istart:istop, jstart:jstop] = x * scale

    # Return the matrix square root.
    return R        

