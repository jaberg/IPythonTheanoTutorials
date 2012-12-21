""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""


import numpy as np
import matplotlib.pyplot as plt

import theano
from theano import tensor
from theano.tensor.nnet.conv import conv2d
import theano.tensor.signal.downsample


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array


def show_filters(x, img_shape, tile_shape, scale_rows_to_unit_interval=True):
    """
    Call matplotlib imshow on the rows of `x`, interpreted as images.

    Parameters:
    x          - a matrix with T rows of P columns
    img_shape  - a (height, width) pair such that `height * width == P`
    tile_shape - a (rows, cols) pair such that `rows * cols == T`
    """
    out = tile_raster_images(x, img_shape, tile_shape, (1, 1),
            scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    plt.imshow(out, cmap=plt.cm.gray, interpolation='nearest')
#    plt.show()


def hinge(margin):
    """Return elementwise hinge loss of margin ndarray"""
    return np.maximum(0, 1 - margin)


def ova_svm_prediction(W, b, x):
    """
    Return a vector of M integer predictions

    Parameters:
    W : weight matrix of shape (N, L)
    b : bias vector of shape (L,)
    x : feature vector of shape (M, N)
    """
    return np.argmax(np.dot(x, W) + b, axis=1)


def ova_svm_cost(W, b, x, y1):
    """
    Return a vector of M example costs using hinge loss

    Parameters:
    W : weight matrix of shape (N, L)
    b : bias vector of shape (L,)
    x : feature vector of shape (M, N)
    y1: +-1 labels matrix shape (M, L)
    """
    # -- one vs. all linear SVM loss
    margin = y1 * (np.dot(x, W) + b)
    cost = hinge(margin).mean(axis=0).sum()
    return cost


def tanh_layer(V, c, x):
    """
    Return layer output matrix of shape (#examples, #outputs)

    Parameters:
    V : weight matrix of shape (#inputs, #outputs)
    c : bias vector of shape (#outputs,)
    x : feature matrix of shape (#examples, #inputs)
    """
    return np.tanh(np.dot(x, V) + c)


def mlp_prediction(V, c, W, b, x):
    h = tanh_layer(V, c, x)
    return ova_svm_prediction(W, b, h)


def mlp_cost(V, c, W, b, x, y1):
    h = tanh_layer(V, c, x)
    return ova_svm_cost(W, b, h, y1)


def theano_fbncc(img4, img_shp, filters4, filters4_shp,
        remove_mean=True, beta=1e-8, hard_beta=True,
        shift3=0, dtype=None, ret_shape=False):
    """
    Channel-major filterbank normalized cross-correlation

    For each valid-mode patch (p) of the image (x), this transform computes

    p_c = (p - mean(p)) if (remove_mean) else (p)
    qA = p_c / sqrt(var(p_c) + beta)           # -- Coates' sc_vq_demo
    qB = p_c / sqrt(max(sum(p_c ** 2), beta))  # -- Pinto's lnorm

    There are two differences between qA and qB:

    1. the denominator contains either addition or max

    2. the denominator contains either var or sum of squares

    The first difference corresponds to the hard_beta parameter.
    The second difference amounts to a decision about the scaling of the
    output, because for every function qA(beta_A) there is a function
    qB(betaB) that is identical, except for a multiplicative factor of
    sqrt(N - 1).

    I think that in the context of stacked models, the factor of sqrt(N-1) is
    undesirable because we want the dynamic range of all outputs to be as
    similar as possible. So this function implements qB.

    Coates' denominator had var(p_c) + 10, so what should the equivalent here
    be?
    p_c / sqrt(var(p_c) + 10)
    = p_c / sqrt(sum(p_c ** 2) / (108 - 1) + 10)
    = p_c / sqrt((sum(p_c ** 2) + 107 * 10) / 107)
    = sqrt(107) * p_c / sqrt((sum(p_c ** 2) + 107 * 10))

    So Coates' pre-processing has beta = 1070, hard_beta=False. This function
    returns a result that is sqrt(107) ~= 10 times smaller than the Coates
    whitening step.

    """
    if dtype is None:
        dtype = img4.dtype

    beta = tensor.as_tensor_variable(beta).astype(dtype)
    shift3 = tensor.as_tensor_variable(shift3).astype(dtype)

    if filters4.dtype != img4.dtype:
        raise TypeError('dtype mistmatch', (img4.dtype, filters4.dtype))
    if dtype != img4.dtype:
        raise TypeError('dtype mistmatch', (dtype, filters4.dtype))

    # -- kernel Number, Features, Rows, Cols
    kN, kF, kR, kC = filters4_shp

    # -- patch-wise sums and sums-of-squares
    box_shp = (1, kF, kR, kC)
    box = tensor.addbroadcast(theano.shared(np.ones(box_shp, dtype=dtype)), 0)
    p_sum = conv2d(img4, box, image_shape=img_shp, filter_shape=box_shp)
    p_mean = 0 if remove_mean else p_sum / (kF * kR * kC)
    p_ssq = conv2d(img4 ** 2, box, image_shape=img_shp, filter_shape=box_shp)

    # -- this is an important variable in the math above, but
    #    it is not directly used in the fused lnorm_fbcorr
    # p_c = x[:, :, xs - xs_inc:-xs, ys - ys_inc:-ys] - p_mean

    # -- adjust the sum of squares to reflect remove_mean
    p_c_sq = p_ssq - (p_mean ** 2) * (kF * kR * kC)
    assert p_c_sq.dtype == dtype
    if hard_beta:
        p_div2 = tensor.maximum(p_c_sq, beta)
    else:
        p_div2 = p_c_sq + beta

    assert p_div2.dtype == dtype
    p_scale = tensor.as_tensor_variable(1.0).astype(dtype) / tensor.sqrt(p_div2)
    assert p_scale.dtype == dtype

    # --
    # from whitening, we have a shift and linear transform (P)
    # for each patch (as vector).
    #
    # let m be the vector [m m m m] that replicates p_mean
    # let a be the scalar p_scale
    # let x be an image patch from s_imgs
    #
    # Whitening means applying the affine transformation
    #   (c - M) P
    # to contrast-normalized patch c = a (x - m),
    # where a = p_scale and m = p_mean.
    #
    # We also want to extract features in dictionary
    #
    #   (c - M) P
    #   = (a (x - [m,m,m]) - M) P
    #   = (a x - a [m,m,m] - M) P
    #   = a x P - a [m,m,m] P - M P
    #

    Px = conv2d(img4, filters4[:, :, ::-1, ::-1],
            image_shape=img_shp,
            filter_shape=filters4_shp,
            border_mode='valid')

    s_P_sum = filters4.sum(axis=[1, 2, 3])
    Pmmm = p_mean * s_P_sum.dimshuffle(0, 'x', 'x')
    if shift3:
        s_PM = (shift3 * filters4).sum(axis=[1, 2, 3])
        z = p_scale * (Px - Pmmm) - s_PM.dimshuffle(0, 'x', 'x')
    else:
        z = p_scale * (Px - Pmmm)

    assert z.dtype == img4.dtype
    z_shp = (img_shp[0], kN, img_shp[2] - kR + 1, img_shp[3] - kC + 1)
    if ret_shape:
        return z, z_shp
    else:
        return z


_theano_fn_cache = {}
def _fbncc(img4, img4_shp, kern4, kern4_shp):
    """
    Return filterbank normalized cross-correlation

    Output has shape
    (#images, #kernels, #rows - #height + 1, #cols - #width + 1)

    See `theano_fbncc` for full documentation of transform.

    Parameters:
    img4 - images tensor of shape (#images, #channels, #rows, #cols)
    kern4 - kernels tensor of shape (#kernels, #channels, #height, #width)

    """
    n_examples, n_channels, n_rows, n_cols = img4.shape
    n_filters, n_channels2, height, width = kern4.shape
    key = ('fbncc', img4.shape, kern4.shape, img4.dtype, kern4.dtype)
    if n_channels != n_channels2:
        raise ValueError('n_channels must match in images and kernels')

    if key not in _theano_fn_cache:
        s_i = theano.tensor.tensor(
                dtype=img4.dtype,
                broadcastable=(
                    n_examples == 1,
                    n_channels == 1,
                    n_rows == 1,
                    n_cols == 1))
        s_k = theano.tensor.tensor(
                dtype=kern4.dtype,
                broadcastable=(
                    n_filters == 1,
                    n_channels == 1,
                    n_rows == 1,
                    n_cols == 1))
        s_y = theano_fbncc(
                s_i, img4.shape,
                s_k, kern4.shape,
                )

        f = theano.function([s_i, s_k], s_y)
        _theano_fn_cache[key] = f
    else:
        f = _theano_fn_cache[key]
    return f(img4, kern4)
_fbncc.__theano_op__ = theano_fbncc

def fbncc(img4, kern4):
    # this call is done in this funny way so that PyAutoDiff can work
    return _fbncc(img4, img4.shape, kern4, kern4.shape)


def max_pool_2d_2x2(img4):
    key = ('maxpool', img4.shape, img4.dtype)

    n_examples, n_channels, n_rows, n_cols = img4.shape

    if key not in _theano_fn_cache:
        s_i = theano.tensor.tensor(
                dtype=img4.dtype,
                broadcastable=(
                    n_examples == 1,
                    n_channels == 1,
                    n_rows == 1,
                    n_cols == 1))
        s_y = theano.tensor.signal.downsample.max_pool_2d(
                s_i, (2, 2), ignore_border=False)
        f = theano.function([s_i], s_y)
        _theano_fn_cache[key] = f
    else:
        f = _theano_fn_cache[key]
    return f(img4)


# N.B. if kwargs are added to fbncc they must match theano_fbncc exactly
#      for PyAutoDiff to work.
max_pool_2d_2x2.__theano_op__ = (lambda img4:
    theano.tensor.signal.downsample.max_pool_2d(img4, (2, 2), False))


def contrast_normalize(patches, remove_mean, beta, hard_beta):
    X = patches
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')
    if remove_mean:
        xm = X.mean(1)
    else:
        xm = X[:,0] * 0
    Xc = X - xm[:, None]
    l2 = (Xc * Xc).sum(axis=1)
    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta
    X = Xc / np.sqrt(div2[:, None])
    return X


def random_patches(images, N, R, C, rng):
    """Return a stack of N uniformly drawn image patches of size (N, channels, R, C)

    Parameters:
    images - 4-tensor of shape (n_images, channels, rows, cols)
    N - integer number of patches to return
    R - rows per patch
    C - columns per patch
    rng - numpy RandomState

    """
    channel_major=True
    if channel_major:
        n_imgs, iF, iR, iC = images.shape
        rval = np.empty((N, iF, R, C), dtype=images.dtype)
    else:
        n_imgs, iR, iC, iF = images.shape
        rval = np.empty((N, R, C, iF), dtype=images.dtype)
    srcs = rng.randint(n_imgs, size=N)
    roffsets = rng.randint(iR - R, size=N)
    coffsets = rng.randint(iC - C, size=N)
    # TODO: this can be done with one advanced index right?
    for rv_i, src_i, ro, co in zip(rval, srcs, roffsets, coffsets):
        if channel_major:
            rv_i[:] = images[src_i, :, ro: ro + R, co : co + C]
        else:
            rv_i[:] = images[src_i, ro: ro + R, co : co + C]
    return rval

MEAN_MAX_NPOINTS = 2000
STD_MAX_NPOINTS = 2000

def mean_and_std(X, remove_std0=False, unbiased=False,
        internal_dtype='float64', return_dtype=None):
    """Return the mean and standard deviation of each column of matrix `X`

    if `remove_std0` is True, then 0 elements of the std vector will be
    switched to 1. This is typically what you want for feature normalization.
    """
    X = X.reshape(X.shape[0], -1)
    npoints, ndims = X.shape

    if npoints < MEAN_MAX_NPOINTS:
        fmean = X.mean(0, dtype=internal_dtype)
    else:
        sel = X[:MEAN_MAX_NPOINTS]
        fmean = np.empty_like(sel[0,:]).astype(internal_dtype)

        np.add.reduce(sel, axis=0, dtype=internal_dtype, out=fmean)

        # -- sum up the features in blocks to reduce rounding error
        curr = np.empty_like(fmean)
        npoints_done = MEAN_MAX_NPOINTS
        while npoints_done < npoints:
            sel = X[npoints_done : npoints_done + MEAN_MAX_NPOINTS]
            np.add.reduce(sel, axis=0, dtype=internal_dtype, out=curr)
            np.add(fmean, curr, fmean)
            npoints_done += MEAN_MAX_NPOINTS
        fmean /= npoints

    if npoints < STD_MAX_NPOINTS:
        fstd = X.std(0, dtype=internal_dtype)
    else:
        sel = X[:MEAN_MAX_NPOINTS]

        mem = np.empty_like(sel).astype(internal_dtype)
        curr = np.empty_like(mem[0,:]).astype(internal_dtype)

        seln = sel.shape[0]
        np.subtract(sel, fmean, mem[:seln])
        np.multiply(mem[:seln], mem[:seln], mem[:seln])
        fstd = np.add.reduce(mem[:seln], axis=0, dtype=internal_dtype)

        npoints_done = MEAN_MAX_NPOINTS
        # -- loop over by blocks for improved numerical accuracy
        while npoints_done < npoints:

            sel = X[npoints_done : npoints_done + MEAN_MAX_NPOINTS]
            seln = sel.shape[0]
            np.subtract(sel, fmean, mem[:seln])
            np.multiply(mem[:seln], mem[:seln], mem[:seln])
            np.add.reduce(mem[:seln], axis=0, dtype=internal_dtype, out=curr)
            np.add(fstd, curr, fstd)

            npoints_done += MEAN_MAX_NPOINTS

        if unbiased:
            fstd = np.sqrt(fstd / max(1, npoints - 1))
        else:
            fstd = np.sqrt(fstd / max(1, npoints))

    if remove_std0:
        fstd[fstd == 0] = 1

    if return_dtype is None:
        return_dtype = X.dtype

    return fmean.astype(return_dtype), fstd.astype(return_dtype)


def patch_whitening_filterbank_X(patches, o_ndim, gamma,
        remove_mean, beta, hard_beta,
        ):
    """
    patches - Image patches (can be uint8 pixels or floats)
    o_ndim - 2 to get matrix outputs, 4 to get image-stack outputs
    gamma - non-negative real to boost low-principle components

    remove_mean - see contrast_normalize
    beta - see contrast_normalize
    hard_beta - see contrast_normalize

    Returns: M, P, X
        M - mean of contrast-normalized patches
        P - whitening matrix / filterbank for contrast-normalized patches
        X - contrast-normalized patches

    """
    # Algorithm from Coates' sc_vq_demo.m

    # -- patches -> column vectors
    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X,
            remove_mean=remove_mean,
            beta=beta,
            hard_beta=hard_beta)

    # -- ZCA whitening (with low-pass)
    print 'patch_whitening_filterbank_X starting ZCA'
    M, _std = mean_and_std(X)
    Xm = X - M
    assert Xm.shape == X.shape
    print 'patch_whitening_filterbank_X starting ZCA: dot', Xm.shape
    C = dot_f64(Xm.T, Xm) / (Xm.shape[0] - 1)
    print 'patch_whitening_filterbank_X starting ZCA: eigh'
    D, V = np.linalg.eigh(C)
    print 'patch_whitening_filterbank_X starting ZCA: dot', V.shape
    P = dot_f32(np.sqrt(1.0 / (D + gamma)) * V, V.T)

    # -- return to image space
    if o_ndim == 4:
        M = M.reshape(patches.shape[1:])
        P = P.reshape((P.shape[0],) + patches.shape[1:])
        X = X.reshape((len(X),) + patches.shape[1:])
    elif o_ndim == 2:
        pass
    else:
        raise ValueError('o_ndim not in (2, 4)', o_ndim)

    print 'patch_whitening_filterbank_X -> done'
    return M, P, X


def fb_whitened_projections(patches, pwfX, n_filters, rseed, dtype):
    """
    pwfX is the output of patch_whitening_filterbank_X with reshape=False

    M, and fb will be reshaped to match elements of patches
    """
    M, P, patches_cn = pwfX
    if patches_cn.ndim != 2:
        raise TypeError('wrong shape for pwfX args, should be flattened',
                patches_cn.shape)
    rng = np.random.RandomState(rseed)
    D = rng.randn(n_filters, patches_cn.shape[1])
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = dot_f32(D, P)
    fb.shape = (n_filters,) + patches.shape[1:]
    M.shape = patches.shape[1:]
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    return M, fb


def fb_whitened_patches(patches, pwfX, n_filters, rseed, dtype):
    """
    pwfX is the output of patch_whitening_filterbank_X with reshape=False

    M, and fb will be reshaped to match elements of patches

    """
    M, P, patches_cn = pwfX
    rng = np.random.RandomState(rseed)
    d_elems = rng.randint(len(patches_cn), size=n_filters)
    D = dot_f64(patches_cn[d_elems] - M, P)
    D = D / (np.sqrt((D ** 2).sum(axis=1))[:, None] + 1e-20)
    fb = dot_f32(D, P)
    fb.shape = (n_filters,) + patches.shape[1:]
    M.shape = patches.shape[1:]
    M = M.astype(dtype)
    fb = fb.astype(dtype)
    if fb.size == 0:
        raise ValueError('filterbank had size 0')
    return M, fb

