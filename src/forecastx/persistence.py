"""모델 저장/불러오기"""
import pickle
import json
import struct
from datetime import datetime
from typing import Optional

MAGIC_BYTES = b'FXM\x01'
FORMAT_VERSION = "1.0"


class ModelPersistence:
    """ForecastX 모델 영속화"""

    @staticmethod
    def save(fxInstance, path: str, metadata: Optional[dict] = None):
        """
        학습된 ForecastX 인스턴스를 .fxm 파일로 저장.

        포맷: MAGIC(4) + metaLen(4) + metaJSON(N) + pickle(M)
        """
        import numpy as np

        meta = {
            'formatVersion': FORMAT_VERSION,
            'forecastxVersion': getattr(fxInstance, 'VERSION', '3.0.0'),
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
        .fxm 파일에서 모델 복원.

        Returns
        -------
        ForecastX 인스턴스 (predict 가능)
        """
        from .forecastx import ForecastX

        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError("올바른 .fxm 파일이 아닙니다.")

            metaLen = struct.unpack('<I', f.read(4))[0]
            metaJson = f.read(metaLen).decode('utf-8')
            meta = json.loads(metaJson)

            state = pickle.load(f)

        fx = ForecastX(verbose=False)
        fx._fittedModels = state.get('fittedModels', {})
        fx.lastResult = state.get('lastResult', None)
        fx._loadedMeta = meta

        return fx

    @staticmethod
    def info(path: str) -> str:
        """모델 파일 정보 조회 (로드 없이)"""
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != MAGIC_BYTES:
                raise ValueError("올바른 .fxm 파일이 아닙니다.")

            metaLen = struct.unpack('<I', f.read(4))[0]
            metaJson = f.read(metaLen).decode('utf-8')
            meta = json.loads(metaJson)

        lines = [
            f"ForecastX Model File v{meta.get('formatVersion', '?')}",
            f"  ForecastX: v{meta.get('forecastxVersion', '?')}",
            f"  생성: {meta.get('createdAt', '?')}",
        ]
        if 'bestModel' in meta:
            lines.append(f"  모델: {meta['bestModel']}")
        if meta.get('metadata'):
            for k, v in meta['metadata'].items():
                lines.append(f"  {k}: {v}")
        return '\n'.join(lines)
