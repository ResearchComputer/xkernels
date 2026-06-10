import torch

from xkernels.ops.comm.reference import all_reduce_reference


def test_comm_reference_is_identity_single_process():
    x = torch.randn(4, 4)
    torch.testing.assert_close(all_reduce_reference([x])[0], x)
