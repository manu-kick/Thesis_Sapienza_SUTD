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
   # Convert cam2worlds_dict tensors into a single stacked tensor
    cam_pos_trains = torch.stack([cam2worlds_dict[k][:3, 3] for k in train_views])  # [V, 3]
    cam_pos_target = torch.stack([cam2worlds_dict[k][:3, 3] for k in test_views])  # [N, 3]

    # Compute absolute differences and sum along spatial dimensions (1 and 2)
    dis = torch.sum(torch.abs(cam_pos_trains[:, None, :] - cam_pos_target[None, :, :]), dim=-1)  # [V, N]

    # Sort distances to get indices of nearest neighbors
    src_idx = torch.argsort(dis, dim=0)  # [V, N], sorted nearest neighbors for each test view

    # Convert indices into proper mapping for train_views
    src_idx = [[train_views[i] for i in src_idx[:, j].tolist()] for j in range(src_idx.shape[1])]



    return src_idx  # This is now a list of lists


def main(args):
    dataset = 'panoptic'
    SEQUENCE = "basketball"  # Set dataset name explicitly
    data_dir = os.path.join("datasets",f"{dataset}_torch")

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
        cam2worlds_dict[cam_id][:3] = cam_pose[6:].view(3, 4)
        # Invert the cam2world matrix because we actually have w2c instead of c2w
        cam2worlds_dict[cam_id] = cam2worlds_dict[cam_id].inverse()

    for cam_id, cam_pose in zip(test_cam_ids, test_scene_data["cameras"]):
        cam2worlds_dict[cam_id] = torch.eye(4, dtype=torch.float32)
        cam2worlds_dict[cam_id][:3] = cam_pose[6:].view(3, 4)
        # Invert the cam2world matrix because we actually have w2c instead of c2w
        cam2worlds_dict[cam_id] = cam2worlds_dict[cam_id].inverse()

     # Compute nearest views using preserved camera IDs

    index = {}
    for scene_data in [test_scene_data]:
        scene_name = scene_data["key"]
        selected_pts = test_cam_ids  # We iterate over test cameras

        for idx, cam_id in enumerate(selected_pts):
            nearest_fixed_views = sorted_test_src_views_fixed(cam2worlds_dict, [cam_id], train_cam_ids)  # Now correctly maps to train_cam_ids
            contexts = tuple(int(x) for x in nearest_fixed_views[0][: args.n_contexts])

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

