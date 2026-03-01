"""Model save/load"""
import json
import pickle
import struct
from datetime import datetime
from typing import Optional

MAGIC_BYTES = b'FXM\x01'
FORMAT_VERSION = "1.0"


class ModelPersistence:
    """Vectrix model persistence"""

    @staticmethod
    def save(fxInstance, path: str, metadata: Optional[dict] = None):
        """
        Save a fitted Vectrix instance to a .fxm file.

        Format: MAGIC(4) + metaLen(4) + metaJSON(N) + pickle(M)
        """

        meta = {
            'formatVersion': FORMAT_VERSION,
            'vectrixVersion': getattr(fxInstance, 'VERSION', '3.0.0'),
            'createdAt': datetime.now().isoformat(),
            'metadata': metadata or {},
        }

        if hasattr(fxInstance, 'lastResult') and fxInstance.lastResult is not None:
            meta['bestModel'] = fxInstance.lastResult.bestModelName

        state = {
            'meta': meta,
            'fittedModels': getattr(fxInstance, '_fittedModels', {}),
            'lastResult': getattr(fxInstance, 'lastResult', None),
        }

        metaBytes = json.dumps(meta, ensure_ascii=False).encode('utf-8')
        stateBytes = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

        with open(path, 'wb') as f:
            f.write(MAGIC_BYTES)
            f.write(struct.pack('<I', len(metaBytes)))
            f.write(metaBytes)
            f.write(stateBytes)

    @staticmethod
    def load(path: str):
        """
        Restore a model from a .fxm file.

        Returns
        -------
        Vectrix instance (ready for predict)
        """
        from .vectrix import Vectrix

        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError("Not a valid .fxm file.")

            metaLen = struct.unpack('<I', f.read(4))[0]
            metaJson = f.read(metaLen).decode('utf-8')
            meta = json.loads(metaJson)

            state = pickle.load(f)

        fx = Vectrix(verbose=False)
        fx._fittedModels = state.get('fittedModels', {})
        fx.lastResult = state.get('lastResult', None)
        fx._loadedMeta = meta

        return fx

    @staticmethod
    def info(path: str) -> str:
        """Query model file info (without loading)"""
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError("Not a valid .fxm file.")

            metaLen = struct.unpack('<I', f.read(4))[0]
            metaJson = f.read(metaLen).decode('utf-8')
            meta = json.loads(metaJson)

        lines = [
            f"Vectrix Model File v{meta.get('formatVersion', '?')}",
            f"  Vectrix: v{meta.get('vectrixVersion', '?')}",
            f"  Created: {meta.get('createdAt', '?')}",
        ]
        if 'bestModel' in meta:
            lines.append(f"  Model: {meta['bestModel']}")
        if meta.get('metadata'):
            for k, v in meta['metadata'].items():
                lines.append(f"  {k}: {v}")
        return '\n'.join(lines)
