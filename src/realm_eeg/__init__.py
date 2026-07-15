"""REALM: robust EEG adaptation through disentanglement and meta-learning."""

from .config import DataConfig, ModelConfig, TrainingConfig
from .diagnostics import PatientProbeResult, patient_identity_linear_probe
from .model import RealmModel, RealmOutput

__all__ = [
    "DataConfig",
    "ModelConfig",
    "PatientProbeResult",
    "RealmModel",
    "RealmOutput",
    "TrainingConfig",
    "patient_identity_linear_probe",
]
__version__ = "0.1.0"
