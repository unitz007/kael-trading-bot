"""Model persistence — save and load trained models.

Models are serialised to disk using ``joblib`` (or ``pickle`` as
fallback) together with a small JSON metadata sidecar that records
the model type, version, training parameters, and evaluation metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata stored alongside every persisted model."""

    model_type: str
    model_version: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] | None = None
    label_values: list[Any] | None = None
    trained_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        return cls(**data)


class ModelPersistence:
    """Save and load trained models with metadata sidecar.

    Parameters
    ----------
    directory:
        Root directory where models are stored.
        Each model gets its own subdirectory:
        ``<directory>/<model_name>/<model_version>/``.
    """

    def __init__(self, directory: str | Path = "models") -> None:
        self.directory = Path(directory)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        metadata: ModelMetadata,
    ) -> Path:
        """Persist a trained model to disk.

        Parameters
        ----------
        model:
            A fitted scikit-learn / XGBoost / LightGBM model.
        model_name:
            Human-readable name (e.g. ``"eurusd_xgboost"``).
        model_version:
            Version string (e.g. ``"v1"`` or ``"2024-01-15T10-30"``).
        metadata:
            Metadata to store alongside the model.

        Returns
        -------
        Path
            The directory where the model was saved.
        """
        model_dir = self.directory / model_name / model_version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model binary
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        logger.info("Saved model binary to %s", model_path)

        # Save metadata JSON
        meta_path = model_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata.to_dict(), indent=2, default=str))
        logger.info("Saved model metadata to %s", meta_path)

        return model_dir

    def load(
        self,
        model_name: str,
        model_version: str,
    ) -> tuple[Any, ModelMetadata]:
        """Load a previously saved model and its metadata.

        Parameters
        ----------
        model_name:
            Name used when saving.
        model_version:
            Version used when saving.

        Returns
        -------
        tuple[model, ModelMetadata]
        """
        model_dir = self.directory / model_name / model_version
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        model = joblib.load(model_path)
        metadata = ModelMetadata.from_dict(json.loads(meta_path.read_text()))
        logger.info("Loaded model from %s", model_dir)
        return model, metadata

    def list_versions(self, model_name: str) -> list[str]:
        """List all available versions for a given model name."""
        model_dir = self.directory / model_name
        if not model_dir.exists():
            return []
        return sorted(
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and (p / "model.joblib").exists()
        )

    def list_models(self) -> list[str]:
        """List all model names that have at least one saved version."""
        if not self.directory.exists():
            return []
        return sorted(
            p.name
            for p in self.directory.iterdir()
            if p.is_dir() and any((p / v / "model.joblib").exists() for v in p.iterdir() if v.is_dir())
        )
