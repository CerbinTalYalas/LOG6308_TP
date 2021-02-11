import numpy as np
import warnings

from numpy.ma.extras import _covhelper, diagflat


def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, allow_masked=True,
             ddof=np._NoValue):
    """
    Return Pearson product-moment correlation coefficients.
    Except for the handling of missing data this function does the same as
    `numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.
    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.
        .. deprecated:: 1.10.0
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises an exception.  Because `bias` is deprecated, this
        argument needs to be treated as keyword only to avoid a warning.
    ddof : _NoValue, optional
        Has no effect, do not use.
        .. deprecated:: 1.10.0
    See Also
    --------
    numpy.corrcoef : Equivalent function in top-level NumPy module.
    cov : Estimate the covariance matrix.
    Notes
    -----
    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.
    """
    msg = 'bias and ddof have no effect and are deprecated'
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Get the data
    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    # Compute the covariance matrix
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask) * 1.
        c = (np.dot(x.T, x.conj(), strict=False) / fact).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T) * 1.
        c = (np.dot(x, x.T.conj(), strict=False) / fact).squeeze()
    # Check whether we have a scalar
    try:
        diag = np.diagonal(c)
    except ValueError:
        return 1
    #
    if xnotmask.all():
        _denom = np.sqrt(np.multiply.outer(diag, diag))
    else:
        _denom = diagflat(diag)
        _denom._sharedmask = False  # We know return is always a copy
        n = x.shape[1 - rowvar]
        if rowvar:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    _x = np.mask_cols(np.vstack((x[i], x[j]))).var(axis=1)
                    _denom[i, j] = _denom[j, i] = np.sqrt(np.multiply.reduce(_x))
        else:
            for i in range(n - 1):
                for j in range(i + 1, n):
                    _x = np.mask_cols(
                            np.vstack((x[:, i], x[:, j]))).var(axis=1)
                    _denom[i, j] = _denom[j, i] = np.sqrt(np.multiply.reduce(_x))
    return c / _denom