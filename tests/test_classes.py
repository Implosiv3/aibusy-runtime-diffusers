"""
A simple test to verify that pytes is working and
the tests are being detected.
"""
import pytest


@pytest.mark.mandatory
def test_classes():
    from aibusy_runtime_diffusers.vae import DiffusersVAE
    from aibusy_runtime_diffusers.unet import DiffusersUNet

    DiffusersVAE(None)
    DiffusersUNet(None)

    assert True