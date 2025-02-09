from model_related import create_img_data_loader, load_classify_model, classify_model_runner_and_save
from weight_assign import WaterBirdImgDistanceWeight, CelebAImgDistanceWeight
from gradcam import GradCAMER
import argparse

def get_weight_assigner(dataset_name: str, **kwargs):
    match dataset_name:
        case 'WaterBird':
            class_name_id_pair = kwargs.get('class_name_id_pair')
            meta_data_path = kwargs.get('meta_data_path')
            img_src_dir = kwargs.get('img_src_dir')
            return WaterBirdImgDistanceWeight(
                dataset_name,
                class_name_id_pair,
                meta_data_path,
                img_src_dir,
            )

        case 'CelebA':
            class_name_id_pair = kwargs.get('class_name_id_pair')
            meta_data_path = kwargs.get('meta_data_path')
            img_src_dir = kwargs.get('img_src_dir')
            return CelebAImgDistanceWeight(
                dataset_name,
                class_name_id_pair,
                meta_data_path,
                img_src_dir,
            )

def parse_args():
    parser = argparse.ArgumentParser()
    # Dataset loader
    parser.add_argument("--img_data_dir", type = str)
    parser.add_argument("--csv_path", type=str)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_name", type=str, default='')

    # Model loader
    parser.add_argument("--model_path", type = str)

    # Cos distance Weight assigner
    parser.add_argument("--add_cos_sim_weights", action="store_true")
    parser.add_argument("--class_name", type=str, default='')
    parser.add_argument("--class_id", type=str, default='')
    parser.add_argument("--meta_data_path", type=str, default='')
    parser.add_argument("--img_src_dir", type=str, default='')

    # GradCAM
    parser.add_argument("--gradcam", action="store_true")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # Load model
    model = load_classify_model(args.model_path)
    
    # Generate dataloader
    data_loader = create_img_data_loader(
        img_data_dir=args.img_data_dir,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_name=args.dataset_name,
    )

    # Run the classify model and save result
    without_class = True if 'without_class' in str(args.img_data_dir) else False

    # Set weight assigner
    weight_assigner = None

    # Prepare gradCAM
    gradCAMer = None
    if args.gradcam:
        gradCAMer = GradCAMER(
            model,
            args.img_data_dir,
        )

    if args.add_cos_sim_weights:
        weight_assigner = get_weight_assigner(
            args.dataset_name,
            class_name_id_pair=(
                args.class_name,
                args.class_id
            ),
            meta_data_path=args.meta_data_path,
            img_src_dir=args.img_src_dir,
        )

    classify_model_runner_and_save(
        model,
        data_loader,
        args.csv_path,
        without_class,
        weight_assigner,
        gradCAMer,
    )
