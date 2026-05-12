import torch
from operators.fft import fft2c, ifft2c


class SingleCoilCartesianOperator:
    """
    Single-coil Cartesian MRI forward and adjoint operator.

    Forward model:
        y = M * F(x)

    Adjoint model:
        x_zf = F^{-1}(M * y)

    This operator assumes centered FFT convention.
    """

    def __init__(self, mask):
        """
        Parameters
        ----------
        mask : torch.Tensor or array-like
            Binary Cartesian undersampling mask.
            Expected shape: (..., H, W) or (H, W)
        """
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask)

        self.mask = mask.float()

    def to(self, device):
        """
        Move mask to device.
        """
        self.mask = self.mask.to(device)
        return self

    def forward(self, x):
        """
        Apply forward MRI operator.

        Parameters
        ----------
        x : torch.Tensor
            Image-domain tensor, shape (..., H, W)

        Returns
        -------
        torch.Tensor
            Undersampled k-space.
        """
        mask = self.mask.to(x.device)
        kspace = fft2c(x)
        return mask * kspace

    def adjoint(self, y):
        """
        Apply adjoint MRI operator.

        Parameters
        ----------
        y : torch.Tensor
            Undersampled k-space, shape (..., H, W)

        Returns
        -------
        torch.Tensor
            Zero-filled complex image reconstruction.
        """
        mask = self.mask.to(y.device)
        return ifft2c(mask * y)

    def normal(self, x):
        """
        Apply A^H A.
        """
        return self.adjoint(self.forward(x))
