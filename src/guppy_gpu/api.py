from typing import no_type_check

from guppylang.decorator import guppy
from guppylang.std.builtins import array, comptime, nat, owned
from guppylang.std.platform import panic

from guppy_gpu.decorator import gpu, gpu_module

######################################
#   GPU DECODER LIBRARY INTERFACE    #
######################################


@gpu_module("cuda-q-qec", None)
class Decoder:
    @gpu
    @no_type_check
    def enqueue_syndromes_ui64(
        self: "Decoder", decoder_id: int, syndrome_size: int, syndrome: int, tag: int
    ) -> None:
        """
        Enqueue a syndrome for decoding.

        Args:
            decoder_id (int): The ID of the decoder to use.
            syndrome_size (int): The size of the syndrome (in bits). Must be <= 64.
            syndrome (int): The bit-packed syndrome to enqueue. The least significant
                            bit (syndrome & 1) is the first bit of the syndrome.
                            The last valid bit is syndrome_size - 1.
            tag (int): The tag to use for the syndrome (logging only).
        """

    @gpu
    def reset_decoder_ui64(self: "Decoder", decoder_id: int) -> None:
        """
        Reset the decoder. Clears any queued syndromes and resets corrections to 0.

        Args:
            decoder_id (int): The ID of the decoder to reset.
        """

    @gpu
    @no_type_check
    def get_corrections_ui64(
        self: "Decoder", decoder_id: int, return_size: int, reset: int
    ) -> int:
        """
        Get the corrections for a given decoder.

        Args:
            decoder_id (int): The ID of the decoder to use.
            return_size (int): The number of bits to return (in bits). Must be <= 64.
            reset (int): Whether to reset the decoder corrections after retrieving them.

        Returns:
            int: The corrections (detected bit flips) for the given decoder.
        """


######################################
#          UTILITY FUNCTIONS         #
######################################


@guppy
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
def unpack_int(data: int, N: nat @ comptime) -> "array[bool, N]":
    """Unpack an integer (assuming big-endian) into a bool array of size N."""
    return array(bool(1 & (data >> (N - n - 1))) for n in range(N))
