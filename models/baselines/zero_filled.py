import torch
import torch.nn as nn


class ZeroFilledReconstructor(nn.Module):
    """
    Zero-filled MRI reconstruction baseline.

    Given undersampled k-space y and an MRI operator A,
    the reconstruction is computed using the adjoint:

        x_zf = A^H(y)

    This is not a learned model. It is used as a baseline.
    """

    def __init__(self, return_magnitude=True):
        super().__init__()
        self.return_magnitude = return_magnitude

    def forward(self, y, operator):
        """
        Parameters
        ----------
        y : torch.Tensor
            Undersampled k-space.

        operator : object
            MRI forward operator with an adjoint method.

        Returns
        -------
        torch.Tensor
            Zero-filled reconstruction.
        """
        x_zf = operator.adjoint(y)

        if self.return_magnitude:
            x_zf = torch.abs(x_zf)

        return x_zf


def zero_filled_reconstruction(y, operator, return_magnitude=True):
    """
    Functional zero-filled reconstruction.

    Parameters
    ----------
    y : torch.Tensor
        Undersampled k-space.

    operator : object
        MRI operator with adjoint method.

    return_magnitude : bool
        If True, return magnitude image.

    Returns
    -------
    torch.Tensor
        Zero-filled reconstruction.
    """
    x_zf = operator.adjoint(y)

    if return_magnitude:
        x_zf = torch.abs(x_zf)

    return x_zf
