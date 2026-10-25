"""Compatibility CLI for the three-stage pooled layerwise BP control."""
import sys
from craft_poisons_transfer import main as run_attack


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if '--feature-mode' not in arguments:
        arguments += ['--feature-mode', 'pool-layerwise']
    if '--mode' not in arguments:
        arguments += ['--mode', 'mean']
    return run_attack(arguments)


if __name__ == '__main__':
    main()
