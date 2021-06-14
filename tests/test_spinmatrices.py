from pyronn_torch import ConeBeamProjector
from pathlib import Path
import xmltodict
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
gpu = torch.device('cuda:0')


def spin_matrices_from_xml(path_to_projection_matrices: str) -> np.ndarray:
    """loads projection matrices in shape (n, 3, 4) from siemens-spin projection matrices in xml"""
    with open(path_to_projection_matrices) as fd:
        mats = xmltodict.parse(fd.read())['hdr']['ElementList']['PROJECTION_MATRICES']
        proj_mat = np.asarray([np.array(mats[k].split(" "), order='C').reshape((3, 4)) for k in mats.keys()])
        logging.info(f"loaded matrices in shape{proj_mat.shape}")
        assert proj_mat.shape[1:] == (3, 4)  # e.g. should be (400, 3, 4)
    return proj_mat.astype(np.float32)


path_to_matrices = Path(r"tests/SpinProjMatrix.xml")
assert path_to_matrices.is_file()
mock_volume = np.zeros((512, 512, 512), dtype=np.float32)
mock_volume[100:400, 200:300, 100:400] = 1.  # define a cube

volume = torch.as_tensor(mock_volume, dtype=torch.float32, device=gpu)
matrices = spin_matrices_from_xml(path_to_matrices)

projector = ConeBeamProjector(volume_shape=(512, 512, 512),
                              volume_spacing=(0.313, 0.313, 0.313),
                              volume_origin=(159.756, 159.756, 159.756),  # [(512 * 0.313) - .5]
                              projection_shape=(400, 976, 976),  # we have 400 matrices
                              projection_spacing=(0.305, 0.305),  # from spin geometry
                              projection_origin=(0, 0),  # i guess this is just for moving the detector in-plane
                              projection_matrices=matrices)

# check if the source points make sense
sp = projector._source_points.cpu().numpy()
logging.info(f"rotation axis is {np.argmin(np.std(sp, axis=0))}")

# create forward projections
projections = projector.project_forward(volume)
if np.all(projections.cpu().data.numpy() == 0):  # make sure something hit the detector
    logging.error("projected data is zero! check setup.")

# plotting
hist_out = Path("distribution_check_projector.png")
points_out = Path("source_points.png")

plt.figure()
plt.scatter(sp[:, 0], sp[:, 1])
plt.title("source points from matrices (x-y plane)")
plt.savefig(points_out)

plt.figure()
plt.subplot(121)
plt.hist(projections.cpu().data.numpy().ravel(), bins=100)
plt.title("projection distribution")
plt.subplot(122)
plt.hist(volume.cpu().data.numpy().ravel(), bins=100)
plt.title("volume distribution")
plt.savefig(hist_out)

logging.info(f"saved images to {hist_out.absolute(), points_out.absolute()}")
