"""Fetches the Olist dataset from Kaggle at container startup if it isn't
already present locally.

The dataset is CC-BY-NC-SA-4.0 licensed (non-commercial, share-alike), so
rather than bundle a copy of it into a publicly-pushed Docker image (which
would mean redistributing it ourselves), the deployed container downloads
its own fresh copy directly from Kaggle on first boot, using credentials
supplied as a deployment secret.
"""

from __future__ import annotations

from pathlib import Path

from src.recsys.data_prep import REQUIRED_FILES


def _download_from_kaggle(data_dir: Path) -> None:
    # imported lazily: `import kaggle` authenticates immediately and raises
    # if no credentials are configured, so this must never run when the
    # data is already present (e.g. local dev, or tests with no credentials)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files("olistbr/brazilian-ecommerce", path=str(data_dir), unzip=True)


def ensure_data_available(data_dir: Path) -> None:
    data_dir = Path(data_dir)
    if all((data_dir / filename).exists() for filename in REQUIRED_FILES.values()):
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    _download_from_kaggle(data_dir)
