import os
from functools import partial
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt

classes = {
    "normal": 0,
    "benign": 1,
    "malignant": 2,
}


class usDataset(Dataset):
    def __init__(
        self,
        dataPath="../Dataset_BUSI_with_GT/",
    ):
        self.datapath = dataPath
        self.filenames, self.labels = self.collect_datafiles(dataPath)

        self.trans = T.Compose(
            [
                T.Resize(size=(384, 384)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                # T.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )

    def collect_datafiles(self, datapath):
        filenames = []
        labels = []

        # check whether datapath is a valid directory
        if not os.path.isdir(datapath):
            raise ValueError(f"Invalid directory {datapath}")

        for root, dirs, files in os.walk(datapath):
            for file in files:
                if "_mask" not in file and not file.startswith("."):
                    filenames.append(os.path.join(root, file))
                    subfolder = os.path.relpath(root, datapath)
                    if subfolder in classes.keys():
                        labels.append(int(classes[subfolder]))
                    else:
                        raise ValueError(f"Unknown class {subfolder}")
        return filenames, labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]

        img = Image.open(filename)
        img = self.trans(img)

        gt = np.zeros([len(classes)], dtype=np.int64)
        gt[label] = 1

        gt = torch.tensor(gt, dtype=torch.float32)

        return img, gt


def usDataloader(cfg, datapath="Dataset_BUSI_with_GT/"):

    dataset = usDataset(dataPath=datapath)

    train_size = int(len(dataset) * 0.7)
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, temp_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    val_dataset, test_dataset = random_split(
        temp_dataset, [val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    return train_loader, val_loader, test_loader


# if __name__=="__main__":
#     prev_case=None
#     dataInfo={}
#     testset=[]
#     with open('../data/NIH_X-ray/test_list_jpg.txt','r') as f:
#         content=f.readlines()
#         for c in content:
#             testset.append(c.strip('\n'))

#     trainset=[]
#     valset=[]
#     train_ratio=7/8
#     with open('../data/NIH_X-ray/train_val_list_jpg.txt','r') as f:
#         train_content=f.readlines()
#         trainNum=int(len(train_content)*train_ratio)
#         for i in range(0,trainNum):
#             trainset.append(train_content[i].strip('\n'))
#         for i in range(trainNum,len(train_content)):
#             valset.append(train_content[i].strip('\n'))
#         # for c in content:
#         #     testset.append(c.strip('\n'))
#     # dataInfo['test']=testset
#     dataInfo['meta']={'trainSize':len(trainset),'valSize':len(valset),'testSize':len(testset)}
#     dataInfo['train']=trainset
#     dataInfo['val']=valset
#     dataInfo['test']=testset

#     with open('nih_split_712.json', 'w') as json_file:
#         json.dump(dataInfo, json_file,indent = 4)
