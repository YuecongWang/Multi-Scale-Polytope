"""Framework-independent method definitions and reproducible run metadata."""
import argparse
import json
from pathlib import Path

from experiment_recorder import digest, save_json


# Each definition is (objective, feature mode, use original attack blocks).
# The third field describes attack representation, not victim parameter updates.
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


def resolve_training_options(legacy_end2end, feature_mode, attack_representation=None,
                             victim_training=None, victim_model_mode='eval'):
    """Resolve attack features and victim training as independent choices.

    Historical --end2end selects full victim fine-tuning and, for original
    features, the original block attack. MSP keeps its selected features.
    """
    if legacy_end2end is not None and not isinstance(legacy_end2end, bool):
        raise ValueError('legacy end2end must be true, false, or omitted')
    if feature_mode not in ('original', 'pool', 'flat', 'flat-layerwise', 'pool-layerwise'):
        raise ValueError('Unknown feature mode: ' + str(feature_mode))
    legacy_training = 'end2end' if legacy_end2end else 'linear'
    if victim_training is not None and victim_training not in ('linear', 'end2end'):
        raise ValueError('victim-training must be linear or end2end')
    if (legacy_end2end is not None and victim_training is not None
            and victim_training != legacy_training):
        raise ValueError('Explicit victim-training contradicts legacy --end2end')
    victim_training = victim_training if victim_training is not None else legacy_training
    if attack_representation is None:
        attack_representation = 'blocks' if legacy_end2end and feature_mode == 'original' else 'features'
    if attack_representation not in ('features', 'blocks'):
        raise ValueError('attack-representation must be features or blocks')
    if attack_representation == 'blocks' and feature_mode != 'original':
        raise ValueError('Original attack blocks require feature-mode=original; MSP uses features')
    if victim_model_mode not in ('eval', 'train'):
        raise ValueError('victim-model-mode must be eval or train')
    if victim_training == 'linear' and victim_model_mode != 'eval':
        raise ValueError('Linear victim training requires model mode eval to keep the backbone frozen')
    return dict(attack_representation=attack_representation,
                attack_end2end=attack_representation == 'blocks',
                victim_training=victim_training, victim_model_mode=victim_model_mode,
                end2end=victim_training == 'end2end')


def validate_attack(mode, feature_mode, attack_blocks, net_repeat=1, poison_num=5):
    if net_repeat < 1 or poison_num < 1:
        raise ValueError("net-repeat and poison-num must be positive")
    if feature_mode not in ("original", "pool", "flat", "flat-layerwise", "pool-layerwise"):
        raise ValueError("Unknown feature mode: " + feature_mode)
    if mode not in ("mean", "convex") and not mode.startswith("coeffs_fixed_type_"):
        raise ValueError("Unknown attack objective: " + mode)
    if attack_blocks and feature_mode != "original":
        raise ValueError("Original attack blocks require feature-mode=original; MSP uses features")
    if feature_mode.endswith("layerwise") and mode != "mean":
        raise ValueError("Layerwise comparisons require mode=mean")
    if mode.startswith("coeffs_fixed_type_"):
        number = mode.removeprefix("coeffs_fixed_type_")
        if number not in {str(i) for i in range(1, 11)}:
            raise ValueError("Fixed coefficient type must be 1..10")
        if feature_mode != "original" or attack_blocks or net_repeat != 1 or poison_num != 5:
            raise ValueError("Fixed coefficient attack requires original features, repeat=1, five poisons")


def method_name(mode, feature_mode, attack_blocks):
    for name, definition in METHODS.items():
        if definition == (mode, feature_mode, attack_blocks):
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
