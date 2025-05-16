import os

import torch
from torch.utils.data import Dataset, DataLoader

'''
patient_id;diagnosis_id;coarse;fine;split
'''

fine_dict = {
    0: 0,
    1: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    10: 8,
    11: 9,
    13: 10
}


def load_anno(anno_path):
    anno_dict = {}
    with open(anno_path, "r") as rf:
        head = rf.readline().strip()
        for line in rf.readlines():
            line_split = line.strip().split(";")
            patient_id = line_split[0]
            diagnosis_id = line_split[1]
            coarse = int(line_split[2])
            fine = int(line_split[3])
            split = line_split[4]

            fine = fine_dict[fine]

            if patient_id not in anno_dict:
                anno_dict[patient_id] = {}
            anno_dict[patient_id][diagnosis_id] = [coarse, fine, split]
    return anno_dict


def load_data(base_dir, anno_dict, split=None):
    data = []
    for year in os.listdir(base_dir):
        year_dir = os.path.join(base_dir, year, "pt_files")
        if not os.path.isdir(year_dir):
            continue
        for filename in os.listdir(year_dir):
            ext = filename.split(".")[-1]
            if ext != "pt":
                continue

            file_path = os.path.join(year_dir, filename)
            if not os.path.isfile(file_path):
                continue

            split_txt = "_diagnose_"
            if split_txt not in filename:
                split_txt = "_diagnosis_"
            patient_id, diagnosis_id = filename.strip(".pt").split(split_txt)

            if patient_id not in anno_dict:
                continue
            if diagnosis_id not in anno_dict[patient_id]:
                continue

            annotation = anno_dict[patient_id][diagnosis_id]
            if split is not None and split != annotation[2]:
                continue
            data.append((file_path, annotation[0], annotation[1], annotation[2])) #, patient_id, diagnosis_id))

    return data


class AMCDataset(Dataset):
    def __init__(self, base_dir, anno_path, split="train"):
        self.anno = load_anno(anno_path)
        self.data = load_data(base_dir, self.anno, split)
        self.split = split

    def __len__(self):
        return len(self.data)

    def __str__(self):
        fine_stats = {}
        for data, coarse_gt, fine_gt, split, patient_id, diagnosis_id in self.data:
            if fine_gt not in fine_stats:
                fine_stats[fine_gt] = 0
            fine_stats[fine_gt] += 1

        msg_fine = ""
        msg_fine += "[{} set] {}\n".format(self.split, len(self))
        for k, v in sorted(fine_stats.items()):
            msg_fine += "Fine {}: {}\n".format(k, v)
        msg_fine += "total num of fine classes: {}\n".format(len(fine_stats))

        coarse_stats = {}
        for data, coarse_gt, fine_gt, split, patient_id, diagnosis_id in self.data:
            if coarse_gt not in coarse_stats:
                coarse_stats[coarse_gt] = 0
            coarse_stats[coarse_gt] += 1
        msg_coarse = ""
        msg_coarse += "[{} set] {}\n".format(self.split, len(self))
        for k, v in sorted(coarse_stats.items()):
            msg_coarse += "Coarse {}: {}\n".format(k, v)
        msg_coarse += "total num of coarse classes: {}\n".format(len(coarse_stats))
        return msg_coarse + msg_fine

    def __getitem__(self, idx):
        data, coarse_gt, fine_gt, split = self.data[idx] #, patient_id, diagnosis_id
        data_tensor = torch.load(data)
        if data_tensor.dim() == 3 and data_tensor.shape[0] == 1:
            data_tensor = data_tensor.squeeze(0)
        return data_tensor, torch.tensor([coarse_gt]), torch.tensor([fine_gt]) #, patient_id, diagnosis_id


def identity_collate(batch):
    return batch[0]


if __name__ == "__main__":
    base_dir = "/home/user/data/UJSMB_STLB"
    anno_path = "/home/user/lib/Capstone_2025/dataloader/amc_fine_grained_anno.csv"

    train_dataset = AMCDataset(base_dir, anno_path, split="train")
    val_dataset = AMCDataset(base_dir, anno_path, split="val")
    test_dataset = AMCDataset(base_dir, anno_path, split="test")

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=identity_collate)
    # for data, coarse_gt, fine_gt, patient_id, diagnosis_id in train_loader:
    #     print(data.shape)

    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
