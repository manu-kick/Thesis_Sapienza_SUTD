import torch
import os
from glob import glob
import argparse
from einops import rearrange, repeat
from dataclasses import asdict, dataclass
import json
from tqdm import tqdm
import numpy as np


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


def sorted_test_src_views_fixed(cam2worlds_dict, test_views, train_views):
    """Use fixed source views for testing instead of selecting different src views dynamically."""
    cam_pos_trains = np.stack([cam2worlds_dict[k].cpu().numpy()[:3, 3] for k in train_views])  # [V, 3], V train views
    cam_pos_target = np.stack([cam2worlds_dict[k].cpu().numpy()[:3, 3] for k in test_views])  # [N, 3], N test views

    dis = np.sum(np.abs(cam_pos_trains[:, None] - cam_pos_target[None]), axis=(1, 2))  # [V, N]
    src_idx = np.argsort(dis) # [N, V] sorted nearest neighbors for each test view

    # Ensure indices map correctly to train_views
    src_idx = [train_views[x] for x in src_idx]

    return src_idx  # This is now a list of lists


def main(args):
    dataset = 'panoptic'
    SEQUENCE = "basketball"  # Set dataset name explicitly
    data_dir = os.path.join("datasets",f"{dataset}_torch", SEQUENCE)

    # Load the test and train torch files
    test_file = os.path.join(data_dir, "basketball_test.torch")
    train_file = os.path.join(data_dir, "basketball_train.torch")

    if not os.path.exists(test_file) or not os.path.exists(train_file):
        raise FileNotFoundError("Torch files not found in expected location!")

    test_scene_data = torch.load(test_file)[0]
    train_scene_data = torch.load(train_file)[0]

    # Extract camera IDs explicitly
    test_cam_ids = test_scene_data["cam_ids"].tolist()
    train_cam_ids = train_scene_data["cam_ids"].tolist()

    # Ensure cam IDs are properly sorted
    all_cam_ids = sorted(test_cam_ids + train_cam_ids)

    # Build explicit cam2world dictionary
    cam2worlds_dict = {}

    for cam_id, cam_pose in zip(train_cam_ids, train_scene_data["cameras"]):
        cam2worlds_dict[cam_id] = torch.eye(4, dtype=torch.float32)
        cam2worlds_dict[cam_id][:3, :] = cam_pose[6:].view(3, 4)

    for cam_id, cam_pose in zip(test_cam_ids, test_scene_data["cameras"]):
        cam2worlds_dict[cam_id] = torch.eye(4, dtype=torch.float32)
        cam2worlds_dict[cam_id][:3, :] = cam_pose[6:].view(3, 4)

     # Compute nearest views using preserved camera IDs

    index = {}
    for scene_data in [test_scene_data]:
        scene_name = scene_data["key"]
        selected_pts = test_cam_ids  # We iterate over test cameras

        for idx, cam_id in enumerate(selected_pts):
            nearest_fixed_views = sorted_test_src_views_fixed(cam2worlds_dict, [cam_id], train_cam_ids)  # Now correctly maps to train_cam_ids
            contexts = tuple([int(x)for x in nearest_fixed_views[: args.n_contexts]])
            targets = (cam_id,)
            index[f"{scene_name}_{cam_id:02d}"] = IndexEntry(
                context=contexts,
                target=targets,
            )
            
    # Save index to a JSON file
    out_path = f"assets/evaluation_index_{SEQUENCE}_nctx{args.n_contexts}.json"
    os.makedirs("assets", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({k: None if v is None else asdict(v) for k, v in index.items()}, f)

    print(f"Dumped index to: {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_contexts", type=int, default=2, help="Number of context views")
    params = parser.parse_args()

    main(params)

