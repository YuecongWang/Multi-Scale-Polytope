"""Run the historical end-to-end target range, preserving four-GPU sharding."""
import argparse
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', type=int, choices=range(4))
    parser.add_argument('mode', choices=['convex', 'mean'])
    parser.add_argument('repeat', type=int, nargs='?', default=1)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error('repeat must be positive')
    # Keep the historical 0..36 target range rather than inventing a new protocol.
    for target in range(37):
        if target % 4 == args.gpu:
            command = ['bash', 'launch/attack-end2end-12.sh', str(args.gpu), args.mode,
                       str(target), str(args.repeat)]
            print(subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
