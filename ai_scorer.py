import asyncio
import json
import logging
from typing import Any

import redis.asyncio as redis
import onnxruntime as ort

LOGGER = logging.getLogger("ai_scorer")


class AIScorer:
    def __init__(self, model_path: str, threshold: float, redis_url: str = "redis://localhost:6379/0"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.redis = redis.from_url(redis_url)

    async def run(self):
        while True:
            raw = await self.redis.brpop("candidates")
            cand = json.loads(raw[1])
            p = self._score(cand)
            if p >= self.threshold:
                cand["p_success"] = float(p)
                await self.redis.lpush("signals", json.dumps(cand))

    def _score(self, cand: dict) -> float:
        # placeholder implementation: expects feature vector under cand['features']
        feats = cand.get("features", {})
        vector = [feats.get("spread", 0.0)]
        ort_inputs = {self.session.get_inputs()[0].name: [vector]}
        prob = self.session.run(None, ort_inputs)[0][0][0]
        return float(prob)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    scorer = AIScorer("model.onnx", threshold=0.5)
    asyncio.run(scorer.run())
