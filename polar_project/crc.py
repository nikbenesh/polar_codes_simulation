import numpy as np
from typing import Iterable

# polynomial for CRC-8 (CRC-8-CCITT / poly 0x07 is common)
CRC8_POLY = 0x07

def _bits_to_int(bits: Iterable[int]) -> int:
    """Convert sequence of bits (MSB first) to integer."""
    v = 0
    for b in bits:
        v = (v << 1) | (1 if int(b) else 0)
    return v

def crc8(bits: Iterable[int]) -> int:
    """
    Compute CRC-8 over a sequence of bits (iterable of 0/1).
    Uses polynomial 0x07, initial CRC = 0x00, bit-by-bit processing (MSB first).
    Returns integer 0..255.
    """
    # ensure we operate with Python int accumulator
    crc = 0
    for bit in bits:
        # bring bit to MSB position
        crc ^= (int(bit) & 1) << 7
        # process 8 times
        for _ in range(8):
            if (crc & 0x80) != 0:
                crc = ((crc << 1) & 0xFF) ^ CRC8_POLY
            else:
                crc = (crc << 1) & 0xFF
    return crc  # 0..255

def append_crc(data_bits):
    """
    Append 8-bit CRC to a bit sequence (numpy array or list).
    Returns numpy array of dtype=int (0/1).
    """
    bits = np.asarray(data_bits, dtype=int).tolist()
    crc_val = crc8(bits)
    crc_bits = [(crc_val >> (7 - i)) & 1 for i in range(8)]
    out = np.concatenate([np.asarray(bits, dtype=int), np.asarray(crc_bits, dtype=int)])
    return out

def check_crc(bits_with_crc) -> bool:
    """
    Check CRC in a bit array that contains data bits followed by 8 CRC bits.
    bits_with_crc: iterable of 0/1 (numpy array or list).
    Returns True if CRC matches, False otherwise.
    """
    arr = np.asarray(bits_with_crc, dtype=int).tolist()
    if len(arr) < 8:
        return False
    data_bits = arr[:-8]
    crc_bits = arr[-8:]
    crc_calc = crc8(data_bits)
    crc_val_from_bits = _bits_to_int(crc_bits)
    return int(crc_calc) == int(crc_val_from_bits)
