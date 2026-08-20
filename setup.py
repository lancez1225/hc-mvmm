"""Build script for HC-MVMM.

Builds the two CUDA extensions used by the detector:

  * ``ops.iou3d_nms.iou3d_nms_cuda``    -- 3-D IoU computation and rotated NMS.
  * ``ops.roiaware_pool3d.roiaware_pool3d_cuda`` -- ROI-aware point pooling.

Run ``python setup.py develop`` from the repository root after installing
the Python dependencies listed in ``requirements.txt`` (in particular the
``spconv`` variant matching your local CUDA toolkit).
"""

import os

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CUDAExtension


def make_cuda_ext(name, module, sources):
    """Helper that wraps ``CUDAExtension`` with consistent source paths."""
    return CUDAExtension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), src) for src in sources],
    )


if __name__ == '__main__':
    setup(
        name='hc_mvmm',
        version='1.0.0',
        description=(
            'HC-MVMM: Hierarchical Confidence-Guided Multi-View Multi-Modal '
            '3-D Object Detection.'
        ),
        author='HC-MVMM authors',
        packages=find_packages(exclude=['paper', 'logs', 'outputs', 'checkpoints']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ],
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ],
            ),
        ],
    )
