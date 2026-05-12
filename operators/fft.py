import torch


def fft2c(x):
    """
    Centered 2D FFT with orthonormal normalization.

    Parameters
    ----------
    x : torch.Tensor
        Input image tensor. Can be real or complex.
        Expected shape: (..., H, W)

    Returns
    -------
    torch.Tensor
        Centered k-space tensor.
    """
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    k = torch.fft.fftshift(k, dim=(-2, -1))
    return k


def ifft2c(k):
    """
    Centered 2D inverse FFT with orthonormal normalization.

    Parameters
    ----------
    k : torch.Tensor
        Centered k-space tensor.
        Expected shape: (..., H, W)

    Returns
    -------
    torch.Tensor
        Complex image-domain tensor.
    """
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x
