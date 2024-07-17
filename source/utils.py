import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='perform the training of the network')
    parser.add_argument(
        '--load-state',
        help='loads and evaluates the best state of the network',
        action=argparse.BooleanOptionalAction,
        default=False
    )
    parser.add_argument(
        '--render-game',
        help='enable game rendering',
        action=argparse.BooleanOptionalAction,
        default=False
    )

    args = parser.parse_args()
    return vars(args)
