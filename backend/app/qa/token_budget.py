"""Deterministic, offline token estimates; never provider billing token counts."""

import base64
from functools import lru_cache
import hashlib
from pathlib import Path

from tiktoken import Encoding

# Official cl100k_base table, vendored so fresh/offline workers do not download it.
TABLE_SHA256 = '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'


@lru_cache(maxsize=1)
def _encoding():
    data = (Path(__file__).parent / 'tokenizer/cl100k_base.tiktoken').read_bytes()
    if hashlib.sha256(data).hexdigest() != TABLE_SHA256:
        raise ValueError('The bundled context tokenizer failed its integrity check.')
    ranks = {base64.b64decode(token): int(rank) for token, rank in
             (line.split() for line in data.splitlines() if line)}
    return Encoding(name='papertrail_cl100k_base', mergeable_ranks=ranks,
        pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        special_tokens={})


def context_tokens(text: str) -> int:
    return len(_encoding().encode_ordinary(text))
