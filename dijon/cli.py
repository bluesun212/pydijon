import argparse
from interpreter import Interpreter


def cli():
    # Create a simple CLI to run code from
    parser = argparse.ArgumentParser(description="Dijon interpreter version 1.0")
    parser.add_argument('-f', '--file', required=True, dest='file')
    filename = parser.parse_args().file

    # Run the code from the file
    Interpreter().run_file(filename)
    print()


if __name__ == '__main__':
    cli()
