import os, glob, json
import datetime
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms


class CarOrientationDataset(Dataset):
    def __init__(self, split, subset="all_data"):
        self.split = split

        # Different processing for datasets with no annotations
        if split == "diva":
            self.dataset = pd.read_csv("car_orientation_dataset_diva.csv")
            self.dataset["orientation_class"] = 0
            return

        # Get/Create dataset
        ds_file = "car_orientation_dataset.csv"
        raw_dataset = pd.read_csv(ds_file) if os.path.isfile(ds_file) else self.create_dataset(ds_file)

        # Create class
        raw_dataset["orientation_class"] = raw_dataset["orientation"].apply(self.convert_orientation_to_class)

        # Get correct portion dataset based on split
        self.dataset = self.get_dataset_split(raw_dataset, split)

        # Filter the dataset
        self.dataset = self.filter_dataset(self.dataset, subset)





    def filter_dataset(self, df, subset):
        '''
        Filter the training data

        Training data sources:
            'chi_multi_image_train', 'chi_full_image_train',
            'weichao_FusionCameraActor_2', 'weichao_FusionCameraActor3_4',
            'weichao_FusionCameraActor4_6', 'weichao_FusionCameraActor5',
            'weichao_FusionCameraActor6', 'weichao_MonitorCamera'
        '''
        if subset == "all_data":
            pass
        elif subset == 'weichao':
            df = df[df["source"].isin([
                'weichao_FusionCameraActor_2', 'weichao_FusionCameraActor3_4',
                'weichao_FusionCameraActor4_6', 'weichao_FusionCameraActor5',
                'weichao_FusionCameraActor6', 'weichao_MonitorCamera'
            ])]
        elif subset == 'chi':
            df = df[df["source"].isin(['chi_multi_image_train', 'chi_full_image_train'])]
        elif subset == 'single_car':
            df = df[df["source"].isin([
                'chi_full_image_train', 'weichao_FusionCameraActor_2',
                'weichao_FusionCameraActor3_4', 'weichao_FusionCameraActor4_6',
                'weichao_FusionCameraActor5', 'weichao_FusionCameraActor6'
            ])]
        elif subset == 'class_balance':
            # Get even number from each orientation
            min_bin = min(np.bincount(df["orientation_class"]))
            df = df.groupby('orientation_class').apply(lambda x: x.sample(min_bin))
        else:
            raise ValueError("Incorrect subset type: '{}'".format(subset))

        return df.reset_index(drop=True)





    def get_dataset_split(self, df, split):
        chi_train = df[df["source"].isin(["chi_multi_image_train", "chi_full_image_train"])]
        chi_val = df[df["source"].isin(["chi_multi_image_val", "chi_full_image_val"])]
        weichao_train = []
        weichao_val = []
        for cam in ["weichao_FusionCameraActor_2", "weichao_FusionCameraActor3_4",
                    "weichao_FusionCameraActor4_6", "weichao_FusionCameraActor5",
                    "weichao_FusionCameraActor6", "weichao_MonitorCamera"]:
            weichao_train.append(df[df["source"] == cam].reset_index(drop=True).iloc[:48000])
            weichao_val.append(df[df["source"] == cam].reset_index(drop=True).iloc[48000:])

        if split == "train":
            return pd.concat([chi_train, *weichao_train]).reset_index(drop=True)
        elif split == "val":
            return pd.concat([chi_val, *weichao_val]).reset_index(drop=True)
        elif split == "test":
            return df[df["source"].isin(["EPFL", "WCVP"])].reset_index(drop=True)
        else:
            raise ValueError("Split must be one of {'train', 'val', 'test'}, got {}".format(split))





    def __len__(self):
        if self.split == "train":
            return 50000
        elif self.split == "val":
            return 10000
        else:
            return(len(self.dataset))





    def get_image(self, im_file, bbox=None):
        # Get cached path
        ssd_path = "{}/{}/{}".format(os.path.dirname(os.path.realpath(__file__)), "cached_ims", im_file.replace("/", "_"))

        # Cached image exists
        if os.path.isfile(ssd_path):
            im = Image.open(ssd_path).copy()
        else:
            im = Image.open(im_file).convert('RGB')
            im.save(ssd_path)

        # Create transform list
        transform_list = []

        # Crop around bounding box if necessary
        if bbox != None:
            im = transforms.functional.crop(im, *bbox)

        # Training transforms
        if self.split == "train":
            transform_list += [
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.ColorJitter(.25,.25,.25,.25),
            ]

        # Testing transforms
        if self.split != "train":
            transform_list += [
                transforms.Resize((224,224)),
            ]

        # Train + Test transforms
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        return transforms.Compose(transform_list)(im)





    def convert_orientation_to_class(self, orientation, bins=36):
        return ((np.round(orientation)%360)/(360/bins)).astype(int)





    def __getitem__(self, idx):
        if self.split in ["train", "val"]:
            row = self.dataset.sample().iloc[0]
        else:
            row = self.dataset.iloc[idx]

        bbox = None if not 'bbox' in row else self.bbox_string_to_PIL_list(row['bbox'])

        return {
            'image': self.get_image(row['im_file'], bbox),
            'image_file': row['im_file'],
            'source': row['source'],
            'orientation': row['orientation_class'],
        }




    def bbox_string_to_PIL_list(self, bbox_string):
        bbox = [int(x) for x in bbox_string.split()]
        bbox = [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]]
        bbox = [
            int(bbox[0] - bbox[2]*.2),
            int(bbox[1] - bbox[3]*.2),
            int(bbox[2] * 1.4),
            int(bbox[3] * 1.4),
        ]
        return bbox




    def show(self, idx):
        row = self.dataset.iloc[idx]
        print(row['im_file'])
        bbox = self.bbox_string_to_PIL_list(row['bbox'])
        im = Image.open(row["im_file"]).convert('RGB')
        im = transforms.functional.crop(im, *bbox)
        im = transforms.functional.resize(im, (224,224))
        im.show()





if __name__ == '__main__':
    ds = CarOrientationDataset("diva")
    ds.show(300)
