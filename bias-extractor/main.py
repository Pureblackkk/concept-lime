import os
import argparse
from b2t_runner import B2TRunner
from keyword_collector import KeywordCollector

def parse_args():
    parser = argparse.ArgumentParser()
    # TODO: extractor arguments
    parser.add_argument("--save_keyword", type=bool, default=True)
    parser.add_argument("--save_keyword_dir", type=str, default='./keyword/')
    parser.add_argument("--skip_bt2", type=bool, default=False)
    
    # B2T arguments
    parser.add_argument("--b2t_dir", type=str, default=None)
    parser.add_argument("--b2t_dataset", type=str, default='waterbird', help="dataset")
    parser.add_argument("--b2t_model", type=str, default='best_model_CUB_erm.pth')
    parser.add_argument("--b2t_extract_caption", default=True)
    parser.add_argument("--b2t_save_result", default=True)
    parser.add_argument("--b2t_cal_score", default=False)
    parser.add_argument("--b2t_keyword_res_dir", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Initialize b2t runner and keyword collector
    MyB2TRunner = B2TRunner(args.b2t_dir)
    MyKeywordCollector = KeywordCollector(
        keyword_dir=os.path.join(args.b2t_dir, args.b2t_keyword_res_dir),
        save_dir=args.save_keyword_dir
    )

    # run b2t if not skip
    if not args.skip_bt2:
        MyB2TRunner.run_b2t(
            dataset=args.b2t_dataset,
            model=args.b2t_model,
            extract_caption=args.b2t_extract_caption,
            save_result=args.b2t_save_result,
            cal_score=args.b2t_cal_score
        )

    # Save keyword if necessary
    if args.save_keyword:
        MyKeywordCollector.save_keyword()



    