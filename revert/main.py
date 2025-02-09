import argparse
from revertor import Revertor

def parse_args():
    parser = argparse.ArgumentParser()
    # revertor arguments
    parser.add_argument("--disturb_src_dir", type=str)
    parser.add_argument("--revert_save_dir", type=str)
    
    # Text-to-image arguments
    parser.add_argument("--model_path_or_name", type=str)
    parser.add_argument("--without_class", type=bool, default=True)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--extend_mode", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    my_revetor = Revertor(
        args.disturb_src_dir,
        args.revert_save_dir,
        args.model_path_or_name,
    )

    my_revetor.revert(
        without_class=args.without_class,
        num_process=args.num_process,
        extend_mode=args.extend_mode,
    )