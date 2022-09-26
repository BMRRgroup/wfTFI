import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.optimize import curve_fit

EPS = 1e-6
gamma_bar = 42.577478518

try:
    import cupy as xp
    import cupy as cp
    from cupy.cuda.memory import OutOfMemoryError as CUDA_OutOfMemory

    from numpy.fft import fftn as cfftn
    from numpy.fft import ifftn as cifftn
    from numpy.fft import fftshift as cfftshift
    from numpy.fft import ifftshift as cifftshift

    from cupy.fft import fftn as gfftn
    from cupy.fft import ifftn as gifftn
    from cupy.fft import fftshift as gfftshift
    from cupy.fft import ifftshift as gifftshift

    def fftn(x):
        if isinstance(x, np.ndarray):
            y = cfftn(x)
        elif isinstance(x, cp.ndarray):
            y = gfftn(x)
        return y

    def ifftn(x):
        if isinstance(x, np.ndarray):
            y = cifftn(x)
        elif isinstance(x, cp.ndarray):
            y = gifftn(x)
        return y

    def fftshift(x):
        if isinstance(x, np.ndarray):
            y = cfftshift(x)
        elif isinstance(x, cp.ndarray):
            y = gfftshift(x)
        return y

    def ifftshift(x):
        if isinstance(x, np.ndarray):
            y = cifftshift(x)
        elif isinstance(x, cp.ndarray):
            y = gifftshift(x)
        return y


    def move2cpu(x, xp=cp):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = x
            elif isinstance(x, cp.ndarray):
                y = cp.asnumpy(x)
            return y
        elif xp == np:
            return x

    def move2gpu(x, xp=cp):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = cp.asarray(x)
            elif isinstance(x, cp.ndarray):
                y = x
            return y
        elif xp == np:
            return x

except ModuleNotFoundError:
    import numpy as xp
    from numpy.fft import fftn, ifftn, fftshift, ifftshift
    def move2cpu(x, xp=np):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        return x

    def move2gpu(x, xp=np):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        return x


def conjugate_gradient(A,b,x=None, precond=None, max_iter=512, reltol=1e-2, verbose=False):
    if isinstance(b, np.ndarray):
        xp = np
    else:
        xp = cp

    """
    Implements conjugate gradient method to solve Ax=b for a large matrix A that is not
    computed explicitly, but given by the linear function A. Also we need a preconditioning matrix precond
    """
    if verbose:
        print("Starting conjugate gradient...")
    if x is None:
        x=xp.zeros_like(b)

    if precond is None:
        # cg standard
        r=b-A(x)
        d=r#matshow(r[:,:,80]);colorbar();show()
        rsnew=xp.sum(r.conj()*r).real
        rs0=rsnew

        if verbose:
            print("initial residual: {}".format(rsnew))

        ii=0
        while ((ii<max_iter) and (rsnew>(reltol**2*rs0))):
            ii=ii+1
            Ad=A(d)
            alpha=rsnew/(xp.sum(d.conj()*Ad))
            x=x+alpha*d
            if ii%50==0:
                #every now and then compute exact residual to mitigate
                # round-off errors
                r=b-A(x)
                d=r
            else:
                r=r-alpha*Ad
            rsold=rsnew
            rsnew=xp.sum(r.conj()*r).real
            d=r+rsnew/rsold*d

            if verbose:
                print("{}, residual: {}".format(ii, rsnew))
    else:
        # cg with preconditioning matrix precond with precondA~id
        r = b-A(x)
        r_invM_r_old = (r.conj()*precond(r)).sum()
        res0 = xp.linalg.norm(r)
        p = precond(r)

        if verbose:
            print("initial residual: {}".format(res0))

        ii=0
        while (ii<=max_iter):
            ii=ii+1
            Ap=A(p)
            alpha=r_invM_r_old/(p.conj()*Ap).real.sum()
            x+=alpha*p
            if (ii % 50)==0:
                #every now and then compute exact residual to mitigate
                # round-off errors
                r=b-A(x)
                p = precond(r)
            else:
                r-=alpha*Ap
            res = xp.linalg.norm(r)
            if res<(reltol*res0):
                exit
            r_invM_r_new = (precond(r).conj()*r).sum()
            beta = r_invM_r_new/r_invM_r_old
            r_invM_r_old = r_invM_r_new
            p = precond(r) + beta*p
            if verbose:
                print("step {}: xp.linalg.norm(residual) = {}".format(ii, res))

    return x


def BFRPDF(fieldmap_ppm, B0dir, voxelSize_mm, mask_tissue, max_iter = 20):
    """ Projection onto dipole fields bachground field removal
    """

    fieldmap_ppm = move2gpu(fieldmap_ppm)
    mask_tissue = move2gpu(mask_tissue)

    matrixSize = fieldmap_ppm.shape

    D = get_dipoleKernel_kspace(matrixSize, voxelSize_mm, B0dir)
    mask_bfr = xp.invert(mask_tissue)

    b = mask_bfr * xp.real(ifftn(D * fftn(mask_tissue * fieldmap_ppm))).astype(xp.float32)

    def A(x):
        fB = xp.real(ifftn(D * fftn(mask_bfr * x)))
        lhs = mask_bfr * xp.real(ifftn(D * fftn(mask_tissue * fB)))
        return lhs

    chiBackground_ppm = conjugate_gradient(A, b, max_iter=max_iter).astype(xp.float32)

    backgroundFieldmap_ppm = xp.real(ifftn(D * fftn(chiBackground_ppm))).astype(xp.float32)
    localFieldmap_ppm = (fieldmap_ppm - backgroundFieldmap_ppm).astype(xp.float32) * mask_tissue

    return move2cpu(chiBackground_ppm), move2cpu(backgroundFieldmap_ppm), \
        move2cpu(localFieldmap_ppm)


def fdiff(x, delta=1.0, axis=0):
    if isinstance(x, np.ndarray):
        xp = np
    elif isinstance(x, cp.ndarray):
        xp = cp
    else:
        print('input variable is neither numpy or cupy array')
        return

    gx = xp.ones_like(x)
    gx[:] = (xp.roll(x, 1, axis=axis) - x) / delta
    return gx


def fdiff_hc(x, delta=1.0, axis=0):
    if isinstance(x, np.ndarray):
        xp = np
    elif isinstance(x, cp.ndarray):
        xp = cp
    else:
        print('input variable is neither numpy or cupy array')
        return

    gx = xp.ones_like(x)
    gx[:] = (xp.roll(x, -1, axis=axis) - x) / delta
    return gx


def calculateReverseCoefficients(paddingParams):
    slicing = []
    padding = []
    padding_nopad = []
    for i in range(0, 3):
        x1_pad = 0
        x2_pad = 0
        x1 = paddingParams['paddingVals'][i] - \
            paddingParams['slicingVals'][i].start
        x2 = paddingParams['paddedShape'][i] - paddingParams['paddingVals'][i] + \
            paddingParams['originalShape'][i] - paddingParams['slicingVals'][i].stop
        if x1 < 0:
            x1_pad = np.abs(x1)
            x1 = 0
        if x2 > paddingParams['paddedShape'][i]:
            x2_pad = x2 - paddingParams['paddedShape'][i]

        x1_nopad = paddingParams['slicingVals'][i].start
        x2_nopad = paddingParams['originalShape'][i] - \
            paddingParams['slicingVals'][i].stop

        slicing.append(slice(x1,x2))
        padding.append((x1_pad, x2_pad))
        padding_nopad.append((x1_nopad, x2_nopad))

    slicing = tuple(slicing)
    padding = tuple(padding)
    padding_nopad = tuple(padding_nopad)
    return slicing, padding, padding_nopad


def compute_gradient_weights(mag, voxelsizes_mm, percentiles=[1.0, 70.0, 99.0]):
    """Compute 3D gradients of magnitude"""

    # maybe average a bit orthogonal to derivative directions?
    # voxelsizes are not used currently, since the scaling is done per direction to find the 100-percentiles[1] strongest edges
    mag_gradients = xp.array([fdiff(mag, voxelsizes_mm[i], axis=i) for i in
                              range(mag.ndim)])

    for mg in mag_gradients:
        mgabs = xp.abs(mg)
        mgmin, mgthresh, mgmax = xp.percentile(mgabs[mag>0.1], percentiles)
        mgabs = (mgabs - mgmin) / (mgmax - mgmin)
        mgthresh = (mgthresh - mgmin ) / (mgmax - mgmin)
        mgabs = xp.maximum(mgthresh - mgabs, 0.0) / mgthresh
        mg[:] = mgabs

    return mag_gradients

def compute_tfipreconditioner(DataParams):
    """Compute preconditioner for TFI based on r2starmap and mask_tissue

    similar to Liu et al, 2020, Automatic Preconditioner
    """

    mask_tissue = DataParams['tissueMask']
    voxelsize = DataParams['voxelSize_mm']
    b0dir = DataParams['B0dir']
    field_ppm = DataParams['RDF_ppm'].astype(np.float32)

    mask_bgr = np.invert(mask_tissue)
    distancemap = distance_transform_edt(mask_bgr, voxelsize)

    xbgr, _, _ = BFRPDF(move2gpu(field_ppm),
                        b0dir, voxelsize,
                        move2gpu(mask_tissue),
                        max_iter = 10)
    xbgr = move2cpu(xbgr)

    dmin, dmax = 0.0, 100.0
    numbins = 100
    bins = np.linspace(dmin, dmax, num=numbins+1)
    distances = []
    chibgrs = []
    for i in range(numbins):
        sel = np.logical_and(distancemap > bins[i], distancemap <= bins[i+1])
        if sel.sum() > 0:
            distances.append(distancemap[sel].mean())
            chibgrs.append(np.median(np.abs(xbgr[sel])))

    def cubic_decay(r, r0, s0):
        return s0 * 1/(1 + r/r0)**3

    popt, _ = curve_fit(cubic_decay, distances, chibgrs, (45.0, 0.7))

    ptfi = np.zeros_like(field_ppm)
    ptfi_max = 30
    ptfi *= mask_tissue
    ptfi[mask_bgr] = ptfi_max / popt[1] * cubic_decay(distancemap[mask_bgr], *popt)
    ptfi[mask_tissue] = 1

    return ptfi


def TFI_linear(DataParams, Options):
    """
    Linear QSM with a Total Variation as in Preconditioned Total-Field-Inversion, Liu et al, MRM
    """

    psi = move2gpu(DataParams['RDF_ppm'])
    if isinstance(psi, np.ndarray):
        xp = np
    elif isinstance(psi, cp.ndarray):
        xp = cp

    psi = psi.astype(xp.float32)

    voxelSize_mm = DataParams['voxelSize_mm']
    B0dir = DataParams['B0dir']

    max_cg_iter = Options['max_cg_iter']
    max_iter = Options['max_iter']
    reltol_update = Options['reltol_update']
    lamda = Options['regularizationParameter']

    if not isinstance(DataParams['P'], int):
        P = move2gpu(DataParams['P'])
    else:
        P = DataParams['P']

    if 'initChi_ppm' in DataParams and DataParams['initChi_ppm'] is not None:
        x = move2gpu(DataParams['initChi_ppm'])
        y = x / P
    else:
        x = xp.zeros_like(psi)

    W = Options['dataWeighting']
    M = Options['gradWeighting']

    W = move2gpu(W)
    M = move2gpu(M)

    W = xp.asarray(W)
    W2 = W**2

    D = get_dipoleKernel_kspace(psi.shape, voxelSize_mm, B0dir)

    if x is None:
        y = xp.zeros_like(psi)
    else:
        x = xp.asarray(x)
        y = x / P


    for t_outer in range(max_iter):
        # weighted gradient of current solution
        modMGPy = xp.zeros_like(y)
        MGpy = xp.zeros((3, *y.shape), dtype=xp.float32)
        for i in range(3):
            MGpy[i] = (M[i, ...] * fdiff(P * y, axis=i, delta=voxelSize_mm[i]))
            modMGPy += MGpy[i]**2
        modMGPy = 1 / xp.sqrt(modMGPy + P**2 * EPS)

        # compute right-hand side for CG
        DPy = ifftn(D * fftn(P * y)).real
        b = P * ifftn(D * fftn(W2 * (psi - DPy))).real
        for i in range(3):
            b -= lamda * P * fdiff_hc(M[i, ...] * modMGPy * MGpy[i], axis=i, delta=voxelSize_mm[i])

        # set up CG operator at current position
        def A(dy):
            lhs = P * ifftn(D * fftn(W2 * ifftn(D * fftn(P * dy)).real)).real
            for i in range(3):
                MGPdyi = M[i, ...] * fdiff(P * dy, axis=i, delta=voxelSize_mm[i])
                lhs += lamda * P * fdiff_hc(M[i, ...] * modMGPy * MGPdyi, axis=i, delta=voxelSize_mm[i])
            return lhs

        dy = conjugate_gradient(A, b, max_iter=max_cg_iter)

        y += dy

        ynorm = xp.linalg.norm(y)
        dynorm = xp.linalg.norm(dy)

        if Options['verbose']:
            print('Iter: {}, update: {}'.format(t_outer, dynorm/ynorm))

        if dynorm/ynorm < reltol_update:
            break

    return move2cpu(P * y)


def wfTFI(DataParams, Options):
    complexsignal = move2gpu(DataParams['signal'])
    if isinstance(complexsignal, np.ndarray):
        xp = np
    elif isinstance(complexsignal, cp.ndarray):
        xp = cp

    complexsignal = complexsignal
    complexwater = move2gpu(DataParams['water'])
    complexfat = move2gpu(DataParams['fat'])
    voxelSize_mm = move2gpu(DataParams['voxelSize_mm'])
    B0dir = move2gpu(DataParams['B0dir'])
    fieldStrength_T = move2gpu(DataParams['fieldStrength_T'])
    r2star = move2gpu(DataParams['r2star'])
    P = move2gpu(DataParams['P'])
    TE_s = move2gpu(DataParams['TE_s'])

    relAmps = move2gpu(Options['relAmps'])
    max_cg_iter = Options['max_cg_iter']
    max_iter = Options['max_iter']
    reltol_update = Options['reltol_update']
    lamda = Options['regularizationParameter']

    deltaP_Hz = Options['freqs_ppm'] * -fieldStrength_T.item() * gamma_bar

    if isinstance(complexsignal, np.ndarray):
        xp = np
    elif isinstance(complexsignal, cp.ndarray):
        xp = cp

    if 'initChi_ppm' in DataParams and DataParams['initChi_ppm'] is not None:
        x = move2gpu(DataParams['initChi_ppm'])
        y = x / P
    else:
        x = xp.zeros_like(r2star)
        y = xp.zeros_like(r2star)

    W = Options['dataWeighting']
    M = Options['gradWeighting']

    W = move2gpu(W)
    M = move2gpu(M)

    if M is None:
        M = xp.ones(complexwater.shape)
    else:
        M = xp.asarray(M)

    D = get_dipoleKernel_kspace(complexwater.shape, voxelSize_mm, B0dir)
    D = D * 2 * xp.pi * gamma_bar * fieldStrength_T

    factor = (gamma_bar * fieldStrength_T) ** 2 * np.sum(TE_s ** 2) * \
        xp.mean(xp.abs(complexwater))

    if xp.mean(xp.abs(complexfat)).item() != 0:
        factor *= xp.mean(xp.abs(complexfat))

    if lamda is not None:
        lamda *= factor
    ne = complexsignal.shape[3]
    P2 = P ** 2

    # set up complexphasor
    complexphasor_array = xp.zeros(ne, dtype=xp.complex64)
    for i in range(ne):
        complexphasor = 0.0
        for amp, deltaP in zip(relAmps, deltaP_Hz):
            complexphasor_array[i] += amp * xp.exp(2.0j * xp.pi * deltaP * TE_s[i])

    for t_outer in range(max_iter):
        # weighted gradient of current solution
        modMGPy = xp.zeros_like(x, dtype=xp.float32)
        MGpy = xp.zeros((3, *x.shape), dtype=xp.float32)
        for i in range(3):
            MGpy[i] = (M[i, ...] * fdiff(P * y, axis=i, delta=voxelSize_mm[i]))
            modMGPy += MGpy[i]**2
        modMGPy = (1 / xp.sqrt(modMGPy + P**2 * EPS)).astype(xp.float32)

        DPy = ifftn(D * fftn(P * y)).real

        # compute right-hand side for CG
        b = xp.zeros_like(y, dtype=xp.float32)
        WF = xp.zeros_like(complexsignal, dtype=xp.complex64)

        for i in range(ne):
            WF[..., i] = complexwater + complexfat * complexphasor_array[i]
            b += TE_s[i] * P * ifftn(D * fftn((complexsignal[..., i] * WF[..., i].conj() *
                                               xp.exp(-r2star * TE_s[i]) *
                                               xp.exp(-1.0j * TE_s[i] * DPy)).imag)).real

        if lamda is not None:
            for i in range(3):
                b -= lamda * P * fdiff_hc(M[i, ...] * modMGPy * MGpy[i], axis=i, delta=voxelSize_mm[i])

        # set up CG operator at current position
        def A(dy):
            lhs = xp.zeros_like(dy, dtype=xp.float32)
            DPdy = ifftn(D * fftn(P * dy)).real

            for i in range(ne):
                WF2 = WF[..., i] * WF[..., i].conj()
                R = xp.exp(2 * -TE_s[i] * r2star)
                lhs += TE_s[i] ** 2 * P * ifftn(D * fftn(WF2 * R * DPdy)).real

            if lamda is not None:
                for i in range(3):
                    MGPdyi = M[i, ...] * fdiff(P * dy, axis=i, delta=voxelSize_mm[i])
                    lhs += lamda * P * fdiff_hc(M[i, ...] * modMGPy * MGPdyi, axis=i, delta=voxelSize_mm[i])

            return lhs

        dy = conjugate_gradient(A, b, max_iter=max_cg_iter)

        y += dy

        ynorm = xp.linalg.norm(y)
        dynorm = xp.linalg.norm(dy)

        if Options['verbose']:
            print('Iter: {}, update: {}'.format(t_outer, dynorm/ynorm))

        if dynorm/ynorm < reltol_update:
            break

    return move2cpu(P * y)


def get_dipoleKernel_kspace(matrixSize, voxelSize_mm, B0dir, DCoffset=0):

    matrixSize = list(matrixSize)
    voxelSize_mm = list(voxelSize_mm)

    i = xp.linspace(-matrixSize[0]//2, matrixSize[0]//2 - 1, matrixSize[0])
    j = xp.linspace(-matrixSize[1]//2, matrixSize[1]//2 - 1, matrixSize[1])
    k = xp.linspace(-matrixSize[2]//2, matrixSize[2]//2 - 1, matrixSize[2])
    J, I, K = xp.meshgrid(j, i, k)


    dk = [1/(a*b) for a,b in zip(voxelSize_mm, matrixSize)]
    Ki = dk[0].item() * I
    Kj = dk[1].item() * J
    Kk = dk[2].item() * K

    Kz = B0dir[0].item() * Ki + B0dir[1].item() * Kj + B0dir[2].item() * Kk
    K2 = Ki**2 + Kj**2 + Kk**2

    center = K2 == 0
    K2[center] = xp.inf

    D = 1/3 - Kz**2 / K2
    D[center] = DCoffset
    return fftshift(D).astype(xp.float32)


def pad_array3d(arr, padsize, xp=xp):
    '''
    :param arr: numpy array
    :returns: padded array symmetrically with size given in padsize (per dimension)

    '''
    arr = move2gpu(arr, xp)
    arrBig = xp.pad(arr, ((padsize[0], padsize[0]), \
                          (padsize[1], padsize[1]), \
                          (padsize[2], padsize[2])), 'constant', \
                    constant_values = 0)
    return move2cpu(arrBig, xp)


def trim_zeros(arr, margin=0):
    '''
    Trim the leading and trailing zeros from a N-D array.

    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object

    '''
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)]*arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
        start = max(start-1-margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end+1+margin) + 1

        s.append(slice(start,end))
    return arr[tuple(s)], tuple(s)
