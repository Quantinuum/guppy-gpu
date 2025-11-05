"""CUDA-Q QEC Realtime Decoding API in Guppy.

An implementation of the C++ API at:
https://github.com/NVIDIA/cudaqx/blob/7eae5b2/libs/qec/include/cudaq/qec/realtime/decoding.h

This module provides a Guppy interface for GPU-accelerated realtime decoding. It includes:

- A `Decoder` class that wraps GPU-accelerated decoding functionality for processing syndromes and
    retrieving corrections in bit-packed integer format.
- Utility functions for packing and unpacking boolean arrays to/from integers for efficient data
    transfer to GPU decoders.

The module uses the `gpu` and `gpu_module` decorators available from guppy-gpu package to enable GPU
execution of quantum error correction algorithms. All syndrome and correction data is represented as
bit-packed uint64 integers with a maximum size of 64 bits.
"""

from hashlib import md5
from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat, owned
from guppylang.std.platform import panic

from guppy_gpu.decorator import gpu, gpu_module

######################################
#   GPU DECODER LIBRARY INTERFACE    #
######################################


def mkhash(x: bytes) -> int:
    """Make a 32-bit hash from bytes."""
    hsh = md5(x).digest()
    return int.from_bytes(hsh[:4], byteorder="little")


@gpu_module("cudaq-qec", None)
class Decoder:
    """Realtime decoding using Nvidia GPUs on Quantinuum hardware.

    Corresponds to the CUDA-Q QEC Realtime decoding API at
    https://github.com/NVIDIA/cudaqx/blob/7eae5b2/libs/qec/lib/realtime/quantinuum/quantinuum_decoding.h

    The decoder maintains internal state for multiple decoder instances, each identified by a
    unique `decoder_id`. Syndromes are enqueued for processing, and corrections (detected bit flips)
    can be retrieved after decoding.

    Note:
        All syndrome and correction data is represented as bit-packed integers (uint64), with a
        maximum size of 64 bits per operation.

    """

    @gpu(mkhash(b"enqueue_syndromes_ui64"))
    @no_type_check
    def enqueue_syndromes(
        self: "Decoder", decoder_id: int, syndrome_size: int, syndrome: int, tag: int
    ) -> None:
        """Enqueue a syndrome for decoding.

        Args:
            decoder_id (int): The ID of the decoder to use.
            syndrome_size (int): The size of the syndrome (in bits). This must be <= 64.
            syndrome (int): The bit-packed syndrome to enqueue. The least significant bit
                            (i.e. syndrome & 1) is the first bit of the syndrome. The last valid bit
                            is `syndrome_size - 1` (i.e. syndrome & (1 << (syndrome_size - 1)).
            tag (int): The tag to use for the syndrome (logging only).

        """

    @gpu(mkhash(b"get_corrections_ui64"))
    @no_type_check
    def get_corrections(
        self: "Decoder", decoder_id: int, return_size: int, reset: int
    ) -> int:
        """Get the corrections for a given decoder.

        Args:
            decoder_id (int): The ID of the decoder to use.
            return_size (int): The number of bits to return (in bits). This must be <= 64. This is
                               expected to match the number of observables in the decoder. The least
                               significant bit (i.e. return_value & 1) is the first bit of the
                               corrections. The last valid bit is `return_size - 1`.
            reset (int): Whether to reset the decoder corrections after retrieving them.

        Returns:
            int: The corrections (detected bit flips) for the given decoder, based on all of the
                 decoded syndromes since the last time any corrections were reset.

        """

    @gpu(mkhash(b"reset_decoder_ui64"))
    def reset_decoder(self: "Decoder", decoder_id: int) -> None:
        """Reset the decoder. This clears any queued syndromes and resets any corrections back to 0.

        Args:
            decoder_id (int): The ID of the decoder to reset.

        """


######################################
#          UTILITY FUNCTIONS         #
######################################


@guppy
@no_type_check
def pack_int(N: nat @ comptime, data: "array[bool, N]" @ owned) -> int:
    """Pack a bool array into an integer (big-endian)."""
    if N > 64:
        panic("Invalid array size for decoder")

    acc = 0
    for b in data:
        acc = acc << 1
        acc = acc | int(b)
    return acc


@guppy
@no_type_check
def unpack_int(data: int, N: nat @ comptime) -> "array[bool, N]":
    """Unpack an integer (assuming big-endian) into a bool array of size N."""
    return array(bool(1 & (data >> (N - n - 1))) for n in range(N))
