# MRIReconLab

A modular PyTorch framework for accelerated MRI reconstruction using optimization-inspired deep unrolling and adversarial reconstruction models.

## Planned Features

- Single-coil MRI reconstruction
- Multi-coil MRI reconstruction
- Cartesian undersampling
- Radial undersampling
- Spiral sampling
- PROPELLER/BLADE-style sampling
- ISTA-Net / ISTA-Net+
- FISTA-Net variants
- FCSA-Net variants
- GAN-based CS-MRI reconstruction

## Repository Structure

- `datasets/` - dataset loaders
- `sampling/` - masks and trajectories
- `operators/` - MRI forward and adjoint operators
- `models/` - reconstruction networks
- `losses/` - loss functions
- `train/` - training scripts
- `evaluate/` - metrics and testing scripts
