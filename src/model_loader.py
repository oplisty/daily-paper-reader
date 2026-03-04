"""统一模型加载器：按顺序尝试下载源，按重试次数回退。"""

from __future__ import annotations

from contextlib import contextmanager
import os
import time
from typing import Callable, Optional

from sentence_transformers import SentenceTransformer


HUGGINGFACE_ENDPOINT = "https://huggingface.co"
MODELSCOPE_ENDPOINT = "https://modelscope.cn/hf"
_DEFAULT_RETRIES = 3


def _log_default(message: str) -> None:
  print(message, flush=True)


@contextmanager
def _hf_endpoint(endpoint: Optional[str] = None):
  had = "HF_ENDPOINT" in os.environ
  old = os.environ.get("HF_ENDPOINT")
  if endpoint:
    os.environ["HF_ENDPOINT"] = endpoint
  elif had:
    del os.environ["HF_ENDPOINT"]

  try:
    yield
  finally:
    if had:
      os.environ["HF_ENDPOINT"] = old
    elif "HF_ENDPOINT" in os.environ:
      del os.environ["HF_ENDPOINT"]


def load_sentence_transformer(
  model_name: str,
  *,
  device: str,
  retries: int = _DEFAULT_RETRIES,
  log: Callable[[str], None] = _log_default,
  providers: tuple[tuple[str, str], ...] = (
    ("huggingface", HUGGINGFACE_ENDPOINT),
    ("modelscope", MODELSCOPE_ENDPOINT),
  ),
):
  attempts = max(int(retries or _DEFAULT_RETRIES), 1)
  last_err: Exception | None = None

  for round_idx in range(1, attempts + 1):
    for provider_name, endpoint in providers:
      try:
        log(
          f"[INFO] 尝试加载模型（第 {round_idx}/{attempts} 轮）：{model_name}"
          f"（provider={provider_name}，device={device}）"
        )
        with _hf_endpoint(endpoint):
          return SentenceTransformer(model_name, device=device)
      except Exception as e:  # pragma: no cover - 仅异常路径
        last_err = e
        msg = str(e)
        if len(msg) > 260:
          msg = msg[:260]
        log(
          f"[WARN] 模型加载失败（provider={provider_name}，round={round_idx}/{attempts}）："
          f"{msg}"
        )

    if round_idx < attempts:
      wait_seconds = 1
      log(f"[INFO] 重试间隔：{wait_seconds}s")
      time.sleep(wait_seconds)

  if last_err is not None:
    raise last_err
  raise RuntimeError(f"加载模型失败：{model_name}")
