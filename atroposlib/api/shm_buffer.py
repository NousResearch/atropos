import array
import json
import logging
import mmap
import os
import struct
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SHMBufferConfig:
    """
    Control block for Shared Memory Buffer.
    Stored at the beginning of the SHM segment.
    """
    # [Magic (4B) | Version (2B) | ReadIdx (4B) | WriteIdx (4B) | MaxSize (4B) | EntrySize (4B)]
    FORMAT = "4sHIIII"
    SIZE = struct.calcsize(FORMAT)
    MAGIC = b"ATRP"
    VERSION = 1


class ZeroCopySHMBuffer:
    """
    High-performance circular buffer using multiprocessing.shared_memory.
    Eliminates JSON serialization and HTTP overhead for trajectory transport.
    """

    def __init__(
        self,
        name: str,
        size: int = 1000,
        entry_size: int = 4096,  # Max tokens per trajectory
        create: bool = False,
    ):
        self.name = name
        self.max_size = size
        self.entry_size = entry_size
        
        # Total size = Control Block + Data Segment
        self.total_size = SHMBufferConfig.SIZE + (size * entry_size * 4) # 4 bytes per int32 token
        
        try:
            if create:
                # Remove existing if any (OS-level cleanup)
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                except FileNotFoundError:
                    pass
                
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=self.total_size)
                self._init_control_block()
                logger.info(f"Created SHM buffer '{name}' with size {self.total_size} bytes")
            else:
                self.shm = shared_memory.SharedMemory(name=name)
                logger.debug(f"Attached to SHM buffer '{name}'")
                
            self.buf = self.shm.buf
        except Exception as e:
            logger.error(f"Failed to initialize SHM buffer: {e}")
            raise

    def _init_control_block(self):
        struct.pack_into(
            SHMBufferConfig.FORMAT,
            self.buf,
            0,
            SHMBufferConfig.MAGIC,
            SHMBufferConfig.VERSION,
            0,  # ReadIdx
            0,  # WriteIdx
            self.max_size,
            self.entry_size,
        )

    def _get_control(self) -> Tuple[int, int, int, int]:
        magic, version, read_idx, write_idx, max_size, entry_size = struct.unpack_from(
            SHMBufferConfig.FORMAT, self.buf, 0
        )
        if magic != SHMBufferConfig.MAGIC:
            raise ValueError("Invalid SHM Magic")
        return read_idx, write_idx, max_size, entry_size

    def _set_indices(self, read_idx: int, write_idx: int):
        # We only update these two fields
        struct.pack_into("II", self.buf, 6, read_idx, write_idx)

    def write_trajectory(self, tokens: List[int], score: float, metadata: Dict[str, Any] = None):
        """
        Writes a trajectory to the buffer without any Python-side copies.
        """
        read_idx, write_idx, max_size, entry_size = self._get_control()
        
        # Check for overflow
        next_write = (write_idx + 1) % max_size
        if next_write == read_idx:
            logger.warning("SHM Buffer Overflow! Dropping trajectory.")
            return False

        # Calculate offset in data segment
        offset = SHMBufferConfig.SIZE + (write_idx * entry_size * 4)
        
        # Zero-copy write using numpy view
        token_arr = np.array(tokens, dtype=np.int32)
        token_len = min(len(token_arr), entry_size)
        
        # View the SHM as a numpy array for the specific slot
        shm_slot = np.ndarray((entry_size,), dtype=np.int32, buffer=self.buf, offset=offset)
        shm_slot[:token_len] = token_arr[:token_len]
        if token_len < entry_size:
            shm_slot[token_len:] = 0 # Padding
            
        # Update write index
        self._set_indices(read_idx, next_write)
        return True

    def read_next(self) -> Optional[np.ndarray]:
        """
        Reads the next available trajectory as a numpy view (no copy).
        """
        read_idx, write_idx, max_size, entry_size = self._get_control()
        
        if read_idx == write_idx:
            return None # Buffer empty
            
        offset = SHMBufferConfig.SIZE + (read_idx * entry_size * 4)
        
        # Return a view of the memory
        data = np.ndarray((entry_size,), dtype=np.int32, buffer=self.buf, offset=offset)
        
        # Advance read index
        self._set_indices((read_idx + 1) % max_size, write_idx)
        return data

    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink:
            self.shm.unlink()
