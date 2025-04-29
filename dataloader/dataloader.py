import csv
import os
import random

CSV_PATH = "./dataloader/20250218_fine_dataset.csv"

DATASET_PATHS = [
    ["2014", "/data6/KU_SM_Jinsol/2014/process_250108/pt_files"],
    ["2015", "/data6/KU_SM_Jinsol/2015/process_250108/pt_files"],
    ["2016", "/data6/KU_SM_Jinsol/2016/process_250108/pt_files"],
    ["2017", "/data6/KU_SM_Jinsol/2017/process_250108/pt_files"],
    ["2018", "/data6/KU_SM_Jinsol/2018/process_241209/pt_files"],
    ["2019", "/data6/KU_SM_Jinsol/2019/process_241209/pt_files"],
    ["2020", "/data6/KU_SM_Jinsol/2020/process_241209/pt_files"],
    ["2021", "/data6/KU_SM_Jinsol/2021/process_241209/pt_files"],
]

def get_os_walk(base_dir, ext=None):
    ext = [ext] if isinstance(ext, str) else ext

    file_paths = []
    for path, dir, files in os.walk(base_dir):
        for filename in files:
            curr_ext = os.path.splitext(filename)[1]
            if ext is not None and curr_ext not in ext:
                continue
            file_path = os.path.join(path, filename)
            file_paths.append(file_path)
    return file_paths

def read_annotation_from_csv(csv_path):
    annotation_data = {}

    with open(csv_path, 'r') as csvfile:
        csvfile.readline()
        for line in csvfile.readlines():
            line = line.replace("\n", "")
            year, patient_id, diagnosis_id, coarse, fine, split = line.split(";")
            year = str(year)
            patient_id = str(patient_id)
            diagnosis_id = str(diagnosis_id)
            coarse = int(coarse)
            fine = int(fine)

            if year not in annotation_data:
                annotation_data[year] = {}
            if patient_id not in annotation_data[year]:
                annotation_data[year][patient_id] = {}
            annotation_data[year][patient_id][diagnosis_id] = (coarse, fine, split)

    return annotation_data


def _get_data_2023_0_2000(data_dir, year):
    data = []
    file_paths = get_os_walk(data_dir, ext=".pt")
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        patient_id = file_path.split(os.path.sep)[-2]
        diagnosis_id = int(filename.split("diagnosis")[-1].split(".")[0])
        data.append([file_path, year, patient_id, diagnosis_id])
    return data


def _get_data_2023_2000_3000(data_dir, year):
    data = []
    file_paths = get_os_walk(data_dir, ext=".pt")
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        patient_id = "_".join(filename.split("_")[:2])
        diagnosis_id = int(filename.split("diagnosis")[-1].split(".")[0])
        data.append([file_path, year, patient_id, diagnosis_id])
    return data


def _get_data(data_dir, year):
    data = []
    file_paths = get_os_walk(data_dir, ext=".pt")
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        patient_id = "_".join(filename.split("_")[:2])
        diagnosis_id = int(filename.split("diagnose_")[-1].split(".")[0])
        data.append([file_path, year, patient_id, diagnosis_id])
    return data


def count_data(data):
    fine_cnt_dict = {i: 0 for i in range(15)}
    for file_path, d in data:
        if d[1] not in fine_cnt_dict:
            fine_cnt_dict[d[1]] = 0
        fine_cnt_dict[d[1]] += 1
    print("Num Fine-grained", ", ".join(["{}: {}".format(k, v) for k, v in fine_cnt_dict.items()]))
    return fine_cnt_dict


def get_data():
    annotation = read_annotation_from_csv(csv_path=CSV_PATH)

    all_data = []
    for year, base_dir in DATASET_PATHS:
        if "Features_512" in base_dir:
            curr_data = _get_data_2023_0_2000(base_dir, year)
        elif "2023_2000_3000" in base_dir:
            curr_data = _get_data_2023_2000_3000(base_dir, year)
        else:
            curr_data = _get_data(base_dir, year)
        all_data.extend(curr_data)

    # total_data = []
    train_data, val_data = [], []
    for file_path, year, patient_id, diagnosis_id in all_data:
        try:
            coarse, fine, split = annotation[str(year)][str(patient_id)][str(diagnosis_id)]
        except KeyError:
            continue
        # total_data.append([file_path, (coarse, fine)])

        if split == "train":
            train_data.append([file_path, (coarse, fine)])
        elif split == "val":
            val_data.append([file_path, (coarse, fine)])
        else:
            raise AssertionError

    print("Train", end=" ")
    count_data(train_data)
    print("Val", end=" ")
    count_data(val_data)

    return train_data, val_data

if __name__ == "__main__":
    random.seed(103)

    _, _ = get_data()

    import torch
    import torch.nn.functional as F

    fine = torch.Tensor([[0.42, 0.28, 0.33, 0.32, 0.68, 0.71, 0.92, 0.85, 0.44, 0.43, 0.12, 0.23, 0.31, 0.29, 0.41]])
    coarse = torch.Tensor([[0.27, 0.81, 0.38, 0.32]])

    print(F.softmax(coarse))
    print(F.softmax(fine))
    print(F.softmax(torch.Tensor([[0.68, 0.71, 0.92, 0.85]])))

    print(F.cross_entropy(F.softmax(fine), torch.LongTensor([6])))
    print(F.cross_entropy(F.softmax(torch.Tensor([[0.68, 0.71, 0.92, 0.85]])), torch.LongTensor([2])))