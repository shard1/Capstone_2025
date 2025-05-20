import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from utils import *
from dataloader.dataloader_AMC import AMCDataset


def generate_kmeans_prototypes(patch_features, k):
    patch_features_np = patch_features.cpu().numpy()
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(patch_features_np)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float)
    return centroids

if __name__ == '__main__':
    set_seed(103)
    all_patch_features = []
    anno_path = "/home/user/lib/Capstone_2025/dataloader/amc_fine_grained_anno.csv"
    base_dir = "/home/user/data/UJSMB_STLB"
    train_dataset = AMCDataset(base_dir, anno_path, split="train", min_patches=8)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)

    for batch in train_loader:
        data, _, _ = batch
        data = data.squeeze(0)
        all_patch_features.append(data)
    all_patch_features = torch.cat(all_patch_features, dim=0)
    coarse_proto = generate_kmeans_prototypes(all_patch_features, 4)
    fine_proto = generate_kmeans_prototypes(all_patch_features, 11)
    torch.save(coarse_proto, "/home/user/lib/Capstone_2025/prototypes/coarse_proto.pt")
    print("Coarse prototypes saved")
    torch.save(fine_proto, "/home/user/lib/Capstone_2025/prototypes/fine_proto.pt")
    print("Fine prototypes saved")