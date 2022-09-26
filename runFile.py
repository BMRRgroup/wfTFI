import pickle
from QSM import QSM
import numpy as np
path2data = './data'

# load example data
with open('./data/spineExample.pickle', 'rb') as f:
    data = pickle.load(f)

# initialize QSM object
obj = QSM(
    data['signal'], # numpy array, nx * ny * nz * nEcho
    data['TE_s'], # list of echo times in seconds, eg [0.0023, 0.0046, 0.0069]
    data['B0dir'], # vector of the B0 field, eg [0,0,1] for B0 along z-direction
    data['voxelSize_mm'], # eg [1.8, 1.8, 1.8] for a 1.8 mm isotropic scan
    data['fieldStrength_T'], # eg 3 for a 3T scanner
    data['water'], # complex water image, numpy array, nx * ny * nz
    data['fat'], # complex fat image, numpy array, nx * ny * nz
    data['R2s_Hz'], # R2* map in Hz, numpy array, nx * ny * nz
    data['fieldmap_ppm'], # field map (B0 map) in ppm, numpy array, nx * ny * nz
)

## ============================== run default ============================== ##
obj.wfTFI() # run QSM algorithm
obj.plot() # visualize inputs and result

## ==================== run with optional parameters ======================= ##
obj.AlgoParams = {
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
    },
    'airSignalThreshold_percent': 10, # threshold for background mask
    'max_cg_iter': 10, # number of conjugate gradients steps per Gauss-Newton step
    'max_iter': 25, # number of maximum Gauss-Newton steps
    'reltol_update': 1e-4, # stopping criteron of relative residual (when update
                           # step becomes small)
    'regularizationParamter': 10 # NEEDS TO BE ADAPTED TO DATA (e.g. l-curve)
}

obj.wfTFI()
obj.plot()
