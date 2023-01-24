# Project: squeezeDetOnKeras
# Filename: subtract_train_from
# Author: Anatoly Khamukhin
# Date: 21.01.2023
# Organisation: personal
# Email: anakhamnoip@gmail.com

import argparse


def subtract_lists(base_file, subtract_file, output_file):
    """Given a two files containing the list of files, make output file with subtracted set list

    Keyword Arguments:
        base_file -- file with full list
        subtract_file -- file with list to exclude from base file
        output_file -- file to store results
    """

    with open(base_file) as base:
        base_files = base.read().splitlines()

    with open(subtract_file) as sub:
        subtract_files = sub.read().splitlines()

    subtracted = list(set(base_files) - set(subtract_files)) #[ f for f in base_files not in subtract_files]

    with open(output_file, 'w') as output:
        output.write("\n".join(subtracted))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Reads files from common list and subtract from them '
                                                 'files in another list. Stores result in third file')
    parser.add_argument("--base", help="File with full list", required=True)
    parser.add_argument("--subtract", help="File with list to subtract", required=True)
    parser.add_argument("--output", help="Output file", required=True)

    args = parser.parse_args()

    subtract_lists(args.base, args.subtract, args.output)


