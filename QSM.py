import numpy as np
import funcLib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cbviewer.cbviewer import cbviewer

try:
    import cupy as xp
    from cupyx.scipy.ndimage import sobel
except ModuleNotFoundError:
    import numpy as xp
    from scipy.ndimage import sobel

class QSM():
    def __init__(self, signal, TE_s, B0dir, voxelSize_mm, fieldStrength_T,
                 water, fat, R2star, fieldmap_ppm):
        self._signal = signal
        self._TE_s = TE_s
        self._B0dir = B0dir
        self._voxelSize_mm = voxelSize_mm
        self._fieldStrength_T = fieldStrength_T
        self._water = water
        self._fat = fat
        self._R2star = R2star
        self._fieldmap_ppm = fieldmap_ppm
        self._paddingParams = dict()
        self._consistencyCheck()
        self.AlgoParams = {
            'verbose': False,
            'FatModel': {
                'freqs_ppm': np.array(
                    [-3.8 , -3.4 , -3.1 , -2.68, -2.46, -1.95, -0.5 ,  0.49,  0.59],
                    dtype=np.float32
                ),
                'relAmps': np.array(
                    [0.08991009, 0.58341658, 0.05994006, 0.08491508, 0.05994006,
                     0.01498501, 0.03996004, 0.00999001, 0.05694306], dtype=np.float32
                )
            }
        }

    def _consistencyCheck(self):
        if self._signal.ndim != 4:
            raise ValueError('The signal arrays needs 4 Dimension (x, y, z, echo)')

        if self._signal.shape[-1] != len(self._TE_s):
            raise ValueError('signal echo time dimension and list of echo times need to be same length')

        shape = self._signal.shape[0:3]
        if self._water.shape != shape:
            raise ValueError('water array has not the same dimensions as signal array')

        if self._fat.shape != shape:
            raise ValueError('fat array has not the same dimensions as signal array')

        if self._R2star.shape != shape:
            raise ValueError('R2star array has not the same dimensions as signal array')

        if self._fieldmap_ppm.shape != shape:
            raise ValueError('fieldmap_ppm array has not the same dimensions as signal array')

    def wfTFI(self):
        self.set_tissueMask(self.AlgoParams.get('airSignalThreshold_percent', 20))
        self._setPaddingParams()
        self.calcPrecon()
        self.set_dataWeighting()
        self.set_gradWeighting()
        self.lTFI()
        self._wfTFI_kernel()

    def lTFI(self):
        DataParams = {'P': self._preconditioner,
                      'voxelSize_mm': self._voxelSize_mm,
                      'B0dir': self._B0dir,
                      'RDF_ppm': self._trimPadArray(self._fieldmap_ppm)}

        Options = {'regularizationParameter': self.AlgoParams.get('regularizationParameterInit', 1e-3),
                   'dataWeighting': self._trimPadArray(self._dataWeighting),
                   'gradWeighting': self._trimPadArray(self._gradWeighting),
                   'max_cg_iter': self.AlgoParams.get('max_cg_iter', 10),
                   'max_iter': self.AlgoParams.get('max_iter', 25),
                   'reltol_update': self.AlgoParams.get('reltol_update_lTFI', 0.01),
                   'verbose': self.AlgoParams['verbose']}

        self._initChi = funcLib.TFI_linear(DataParams, Options)

    def _wfTFI_kernel(self):
        DataParams = {'voxelSize_mm': self._voxelSize_mm,
                      'B0dir': self._B0dir,
                      'water': self._trimPadArray(self._water),
                      'fat': self._trimPadArray(self._fat),
                      'fieldStrength_T': self._fieldStrength_T,
                      'r2star': self._trimPadArray(self._R2star),
                      'signal': self._getTrimPadSignal(),
                      'TE_s': self._TE_s,
                      'P': self._preconditioner,
                      'initChi_ppm': self._initChi}

        Options = {'regularizationParameter': self.AlgoParams.get('regularizationParameter', 10),
                   'dataWeighting': self._trimPadArray(self._dataWeighting),
                   'gradWeighting': self._trimPadArray(self._gradWeighting),
                   'max_cg_iter': self.AlgoParams.get('max_cg_iter', 10),
                   'max_iter': self.AlgoParams.get('max_iter', 25),
                   'reltol_update': self.AlgoParams.get('reltol_update', 1e-4),
                   'verbose': self.AlgoParams['verbose'],
                   'freqs_ppm': self.AlgoParams['FatModel']['freqs_ppm'],
                   'relAmps': self.AlgoParams['FatModel']['relAmps']}

        self.susceptibility = funcLib.wfTFI(DataParams, Options)

    def calcPrecon(self):
        DataParams = {'tissueMask': self._trimPadArray(self._mask),
                      'voxelSize_mm': self._voxelSize_mm,
                      'B0dir': self._B0dir,
                      'RDF_ppm':  self._trimPadArray(self._fieldmap_ppm)}

        self._preconditioner = funcLib.compute_tfipreconditioner(DataParams)

    def _setPaddingParams(self):
        paddingParams = self._paddingParams
        paddingParams['originalShape'] = self._signal.shape[0:3]

        array, slicing = funcLib.trim_zeros(self._mask)
        paddingParams['trimmedShape'] = array.shape
        paddingParams['slicingVals'] = slicing

        if 'paddingVals' not in paddingParams:
            paddingParams['paddingVals'] = np.ceil(np.array(paddingParams['trimmedShape']) / 4).astype(np.int32)

        paddingParams['paddedShape'] = tuple(np.array(paddingParams['trimmedShape']) + \
            2 * np.array(paddingParams['paddingVals']))

        revSlicing, revPadding, revPadding_nopad = \
            funcLib.calculateReverseCoefficients(paddingParams)
        paddingParams['reverseSlicingVals'] = revSlicing
        paddingParams['reversePaddingVals'] = revPadding
        paddingParams['reversePaddingValsNoPad'] = revPadding_nopad

    def set_dataWeighting(self):
        MIP = self._getEchoMIP()
        weights = MIP / np.percentile(MIP, 95.0)
        weights[weights > 1] = 1.0
        self._dataWeighting = (weights * self._mask).astype(np.float32)

    def set_gradWeighting(self, method='gradient'):
        MIP = funcLib.move2gpu(self._getEchoMIP())
        if method == 'sobel':
            edges = xp.zeros((3,) + MIP.shape).astype(xp.float32)
            for i in range(3):
                edges[i, ...] = xp.abs(sobel(MIP, axis=i))

            edges /= xp.percentile(edges, 98)
            edges[edges > 1] = 1.0
            edges = (1 - edges)
        elif method == 'gradient':
            edges = funcLib.compute_gradient_weights(
                MIP, self._voxelSize_mm, [1.0, 95.0, 99.0])
        else:
            raise ValueError(f"method {method} not supported")

        edges[edges < 0.1] = 0.1
        edges = funcLib.move2cpu(edges)

        for i in range(3):
            edges[i, ...] = edges[i, ...] * self._mask

        self._gradWeighting = funcLib.move2cpu(edges)

    def plot(self):
        entries = []
        entries.append([np.abs(self._water), {'title': 'water', 'cmap': 'gray'}])
        entries.append([np.abs(self._fat), {'title': 'fat', 'cmap': 'gray'}])
        entries.append([self._R2star, {'title': '$R_2^*$', 'cmap': 'plasma', 'clabel':'[Hz]', 'window': [0, 300] }])

        pctl_lims = [0.3, 99.7]

        fm = self._fieldmap_ppm
        mask = self._mask
        fm_lims = [np.percentile(fm[mask], pctl_lims[0]), np.percentile(fm[mask], pctl_lims[1])]
        entries.append([self._fieldmap_ppm, {'title': 'fieldmap', 'cmap': 'magma', 'clabel':'[ppm]', 'window': fm_lims}])

        chi = self._reverseTrimPadArray(self.susceptibility) * self._mask
        chi_lims = [np.percentile(chi[mask], pctl_lims[0]), np.percentile(chi[mask], pctl_lims[1])]
        entries.append([chi, {'title': 'susceptibility', 'cmap': 'viridis', 'clabel':'[ppm]', 'window': chi_lims}])

        cbviewer(entries)

    def set_tissueMask(self, threshold=5):
        MIP = self._getEchoMIP()
        self._mask = MIP > threshold / 100 * np.max(MIP)

    def _getEchoMIP(self):
        return np.max(np.abs(self._signal), axis=-1)

    def _getTrimPadSignal(self):
        slicing = self._paddingParams['slicingVals']
        padding = self._paddingParams['paddingVals']

        ne = self._signal.shape[-1]
        retSignal = np.zeros((*self._paddingParams['paddedShape'], ne), dtype=np.complex64)

        for ie in range(0, ne):
            tmpsignal = self._signal[..., ie]
            retSignal[..., ie] = funcLib.pad_array3d(tmpsignal[slicing], padding)

        return retSignal

    def _trimPadArray(self, arr):
        slicing = self._paddingParams['slicingVals']
        padding = self._paddingParams['paddingVals']

        if arr.shape == self._paddingParams['paddedShape']:
            return arr
        else:
            if arr.ndim == 3:
                retArr = funcLib.pad_array3d(arr[slicing], padding)
            elif arr.ndim == 4:
                retArr = \
                    np.zeros((3, *self._paddingParams['paddedShape']),
                             dtype=arr.dtype)
                for i in range(3):
                    retArr[i, ...] = funcLib.pad_array3d(arr[i, ...][slicing], padding)

            return retArr

    def _reverseTrimPadArray(self, arr):
        paddingParams = self._paddingParams

        if arr.shape == paddingParams['originalShape']:
            return arr

        retArr = np.pad(arr[paddingParams['reverseSlicingVals']],
                        ((paddingParams['reversePaddingVals'][0][0],
                          paddingParams['reversePaddingVals'][0][1]),
                         (paddingParams['reversePaddingVals'][1][0],
                          paddingParams['reversePaddingVals'][1][1]),
                         (paddingParams['reversePaddingVals'][2][0],
                          paddingParams['reversePaddingVals'][2][1])),
                        'constant', constant_values = 0)
        return retArr
