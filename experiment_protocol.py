"""Framework-independent method definitions and reproducible run metadata."""
import argparse
import json
from pathlib import Path

from experiment_recorder import digest, save_json


METHODS = {
    "CP-Original": ("convex", "original", False),
    "BP-Original": ("mean", "original", False),
    "MSP-Pool": ("mean", "pool", False),
    "MSP-Flat": ("mean", "flat", False),
    "MSP-Flat-Layerwise": ("mean", "flat-layerwise", False),
    "BP-3Layer-Pool-Control": ("mean", "pool-layerwise", False),
    "CP-E2E-Original": ("convex", "original", True),
    "BP-E2E-Original": ("mean", "original", True),
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def validate_attack(mode, feature_mode, end2end, net_repeat=1, poison_num=5):
    if net_repeat < 1 or poison_num < 1:
        raise ValueError("net-repeat and poison-num must be positive")
    if feature_mode not in ("original", "pool", "flat", "flat-layerwise", "pool-layerwise"):
        raise ValueError("Unknown feature mode: " + feature_mode)
    if mode not in ("mean", "convex") and not mode.startswith("coeffs_fixed_type_"):
        raise ValueError("Unknown attack objective: " + mode)
    if end2end and feature_mode != "original":
        raise ValueError("MSP features currently support linear transfer only; end2end would ignore them")
    if feature_mode.endswith("layerwise") and mode != "mean":
        raise ValueError("Layerwise comparisons require mode=mean")
    if mode.startswith("coeffs_fixed_type_"):
        number = mode.removeprefix("coeffs_fixed_type_")
        if number not in {str(i) for i in range(1, 11)}:
            raise ValueError("Fixed coefficient type must be 1..10")
        if feature_mode != "original" or end2end or net_repeat != 1 or poison_num != 5:
            raise ValueError("Fixed coefficient experiment requires original, linear, repeat=1, five poisons")


def method_name(mode, feature_mode, end2end):
    for name, definition in METHODS.items():
        if definition == (mode, feature_mode, end2end):
            return name
    return f"custom-{mode}-{feature_mode}"


def checkpoint_path(directory, template, model):
    """Support the historical %s templates and explicit checkpoint filenames."""
    name = template % model if "%s" in template else template
    return Path(directory) / name


def file_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": digest(path)}


def ensure_protocol(path, protocol):
    """An existing run cannot silently change algorithm, inputs or parameters."""
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            changed = sorted(key for key in set(previous) | set(protocol)
                             if previous.get(key) != protocol.get(key))
            raise ValueError("Run protocol changed (" + ", ".join(changed)
                             + "); choose a new --chk-path/output_root")
    else:
        if path.parent.exists() and any(path.parent.glob("poison_*.pth")):
            raise ValueError("Unrecorded historical poisons exist here; use a new output directory")
        save_json(path, protocol)
