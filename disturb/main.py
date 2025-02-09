import argparse
from disturb import DataDisturbing


def parse_args():
    parser = argparse.ArgumentParser()

    # Disturb
    parser.add_argument("--src_dir", type=str, default='./keyword/')
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default='./data/disturb/')

    # Generate caption prompt based on disturb result
    parser.add_argument("--model_name", type=str, default='Meta-Llama-3-8B-Instruct.Q4_0.gguf')
    parser.add_argument("--generate_prompt", type=bool, default=True)
    parser.add_argument("--prompt_with_class", type=bool, default=False)
    parser.add_argument("--template_type", type=str, default='short')
    parser.add_argument("--shots_num", type=int, default=1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    MyDataDisturbing = DataDisturbing(
        src_dir=args.src_dir,
        save_dir=args.save_dir,
        model_name=args.model_name,
    )
    
    # Disturb and save the result
    MyDataDisturbing.disturbing(
        sample_num = args.sample_num,
        generate_prompt=args.generate_prompt,
        template_type=args.template_type,
        shots_num=args.shots_num,
        prompt_with_class=args.prompt_with_class
    )

