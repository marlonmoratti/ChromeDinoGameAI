import argparse

def parse_arguments():
    parse = argparse.ArgumentParser(description='DinoAI')
    parse.add_argument(
        '--load-state',
        help='loads and evaluates the best state of the algorithm',
        action=argparse.BooleanOptionalAction,
        default=False
    )

    args = parse.parse_args()
    return vars(args)
