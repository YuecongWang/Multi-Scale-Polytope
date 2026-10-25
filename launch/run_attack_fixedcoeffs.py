"""Run five-poison fixed-coefficient controls for targets 0..49.

The transfer wrapper invokes the current craft entrypoint, which already performs
victim evaluation. Historical eval-transfer.sh parses the old log format and must
not be called a second time against the new protocol logs.
"""
import argparse
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpu', type=int, choices=range(4))
    parser.add_argument('type', type=int, choices=range(1, 11))
    args = parser.parse_args(argv)
    mode = 'coeffs_fixed_type_{}'.format(args.type)
    # The defined coefficient vectors have exactly five entries and repeat=1.
    for target in range(50):
        if target % 4 == args.gpu:
            command = ['bash', 'launch/attack-transfer-18.sh', str(args.gpu), mode,
                       str(target), '1', '5']
            print(subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
