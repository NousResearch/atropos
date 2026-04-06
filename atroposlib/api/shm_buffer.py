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

    # [Magic (4B) | Version (4B) | ReadIdx (4B) | WriteIdx (4B) | MaxSize (4B) | EntrySize (4B)]
    FORMAT = "4sIIIII"
    SIZE = struct.calcsize(FORMAT)
    MAGIC = b"ATRP"
    VERSION = 1


class ZeroCopySHMBuffer:
    """
    High-performance circular buffer using multiprocessing.shared_memory.
    Eliminates serialization and HTTP overhead for trajectory transport.
    """

    def __init__(
        self,
        name: str,
        size: int = 1000,
        entry_size: int = 4096,  # Max tokens per trajectory
        instance_id_len: int = 64,
        metadata_len: int = 256,
        create: bool = False,
    ):
        self.name = name
        self.max_size = size
        self.entry_size = entry_size
        self.instance_id_len = instance_id_len
        self.metadata_len = metadata_len

        # Schema: [Score (8) | Len (4) | InstanceID (id_len) | RepID (4) | Meta (meta_len) | Tokens (Size*4)]
        self.slot_size = 8 + 4 + instance_id_len + 4 + metadata_len + (entry_size * 4)

        # Total size = Control Block + Data Segment
        self.total_size = SHMBufferConfig.SIZE + (size * self.slot_size)

        try:
            if create:
                # Remove existing if any (OS-level cleanup)
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    shm.unlink()
                except FileNotFoundError:
                    pass

                self.shm = shared_memory.SharedMemory(
                    name=name, create=True, size=self.total_size
                )
                self.buf = self.shm.buf
                self._init_control_block()
                logger.info(
                    f"Created SHM buffer '{name}' with size {self.total_size} bytes"
                )
            else:
                self.shm = shared_memory.SharedMemory(name=name)
                self.buf = self.shm.buf
                logger.debug(f"Attached to SHM buffer '{name}'")
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

    def _set_read_idx(self, idx: int):
        struct.pack_into("I", self.buf, 8, idx)

    def _set_write_idx(self, idx: int):
        struct.pack_into("I", self.buf, 12, idx)

    def write_trajectory(
        self,
        tokens: List[int],
        score: float,
        instance_id: str = "",
        repetition_id: int = 0,
        metadata: Dict[str, Any] = None,
    ):
        """
        Writes a trajectory and its rich metadata to the buffer.
        """
        read_idx, write_idx, max_size, entry_size = self._get_control()

        # Check for overflow
        next_write = (write_idx + 1) % max_size
        if next_write == read_idx:
            logger.warning("SHM Buffer Overflow! Dropping trajectory.")
            return False

        # Calculate offset in data segment
        offset = SHMBufferConfig.SIZE + (write_idx * self.slot_size)

        # Pack Metadata and Rich attributes
        struct.pack_into("d", self.buf, offset, float(score))

        token_len = min(len(tokens), entry_size)
        struct.pack_into("i", self.buf, offset + 8, token_len)

        id_bytes = instance_id.encode("utf-8")[: self.instance_id_len]
        struct.pack_into(f"{self.instance_id_len}s", self.buf, offset + 12, id_bytes)

        struct.pack_into(
            "i", self.buf, offset + 12 + self.instance_id_len, int(repetition_id)
        )

        meta_json = json.dumps(metadata or {}).encode("utf-8")[: self.metadata_len]
        struct.pack_into(
            f"{self.metadata_len}s",
            self.buf,
            offset + 12 + self.instance_id_len + 4,
            meta_json,
        )

        # Copy tokens via Numpy View directly into SHM slot
        token_offset = offset + 12 + self.instance_id_len + 4 + self.metadata_len
        token_arr = np.array(tokens, dtype=np.int32)
        shm_slot = np.ndarray(
            (entry_size,), dtype=np.int32, buffer=self.buf, offset=token_offset
        )
        shm_slot[:token_len] = token_arr[:token_len]
        if token_len < entry_size:
            shm_slot[token_len:] = 0

        self._set_write_idx(next_write)
        return True

    def read_next(self) -> Optional[Dict[str, Any]]:
        """
        Reads the next available trajectory with its score and metadata.
        """
        read_idx, write_idx, max_size, entry_size = self._get_control()

        if read_idx == write_idx:
            return None  # Buffer empty

        offset = SHMBufferConfig.SIZE + (read_idx * self.slot_size)

        # Unpack Metadata and Rich attributes
        score = struct.unpack_from("d", self.buf, offset)[0]
        token_len = min(struct.unpack_from("i", self.buf, offset + 8)[0], entry_size)

        id_bytes = struct.unpack_from(
            f"{self.instance_id_len}s", self.buf, offset + 12
        )[0]
        instance_id = id_bytes.decode("utf-8", errors="ignore").strip("\x00")

        repetition_id = struct.unpack_from(
            "i", self.buf, offset + 12 + self.instance_id_len
        )[0]

        meta_bytes = struct.unpack_from(
            f"{self.metadata_len}s", self.buf, offset + 12 + self.instance_id_len + 4
        )[0]
        try:
            metadata = json.loads(
                meta_bytes.decode("utf-8", errors="ignore").strip("\x00")
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            metadata = {}

        token_offset = offset + 12 + self.instance_id_len + 4 + self.metadata_len
        tokens_view = np.ndarray(
            (token_len,), dtype=np.int32, buffer=self.buf, offset=token_offset
        )

        self._set_read_idx((read_idx + 1) % max_size)

        return {
            "tokens": tokens_view.tolist(),
            "score": score,
            "instance_id": instance_id,
            "repetition_id": repetition_id,
            "metadata": metadata,
        }

    def close(self, unlink: bool = False):
        self.shm.close()
        if unlink:
            self.shm.unlink()
