"""Evaluate historical result folders, dividing targets 0..49 across four GPUs."""
import argparse
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', type=int, choices=range(4))
    parser.add_argument('poisons_root', help='directory directly containing historical target folders')
    args = parser.parse_args(argv)
    if not Path(args.poisons_root).is_dir():
        parser.error('poisons_root must be an existing directory')
    for target in range(50):
        if target % 4 == args.gpu:
            command = ['bash', 'launch/eval-transfer.sh', str(args.gpu), str(target), args.poisons_root]
            print(subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
