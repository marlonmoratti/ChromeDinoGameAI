import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='DinoAI')
    parser.add_argument(
        '--load-state',
        help='Loads and evaluates the best state of the algorithm',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--render-game',
        help='Enable game rendering',
        action=argparse.BooleanOptionalAction,
        default=False
    )

    args = parser.parse_args()
    return vars(args)
