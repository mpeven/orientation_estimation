import os, glob, json
import datetime
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
pd.set_option('display.width', 200)
from torch.utils.data import Dataset
from torchvision import transforms


class CarOrientationDataset(Dataset):
    def __init__(self, split):
        self.split = split

        # Get/Create dataset
        ds_file = "car_orientation_dataset.csv"
        raw_dataset = pd.read_csv(ds_file) if os.path.isfile(ds_file) else self.create_dataset(ds_file)

        # Create class
        raw_dataset["orientation_class"] = raw_dataset["orientation"].apply(self.convert_orientation_to_class)

        # Get correct portion dataset based on split
        self.dataset = self.get_dataset_split(raw_dataset, split)

        # Filter the dataset so distributions are even
        if split == "train":
            self.dataset = self.filter_dataset(self.dataset)





    def filter_dataset(self, df):
        # Get even number of chi/weichao
        df1 = df[df["source"].isin(["chi_multi_image_train", "chi_full_image_train"])]
        df2 = df[~df["source"].isin(["chi_multi_image_train", "chi_full_image_train"])]
        min_df = min([len(df1), len(df2)])
        df = pd.concat([df1.sample(min_df), df2.sample(min_df)]).reset_index(drop=True)

        # Get even number from each orientation
        min_bin = min(np.bincount(df["orientation_class"]))
        df = df.groupby('orientation_class').apply(lambda x: x.sample(min_bin)).reset_index(drop=True)

        return df





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
            return df[df["source"] == "EPFL"].reset_index(drop=True)
        else:
            raise ValueError("Split must be one of {'train', 'val', 'test'}, got {}".format(split))





    def create_dataset(self, save_file):
        dataset = []

        #### Get Chi's data
        for split in ["train", "val"]:
            chi_multi_images = sorted(glob.glob("/hdd/Datasets/ChiCars/car_multi/{}/*/*.png".format(split)))
            chi_full_images  = sorted(glob.glob("/hdd/Datasets/ChiCars/car_full/{}/*/*.png".format(split)))
            for source, im_paths in [("chi_multi_image", chi_multi_images), ("chi_full_image", chi_full_images)]:
                for im_path in tqdm(im_paths, ncols=115, desc="Getting Chi's '{}' data".format(source)):
                    dataset.append({
                        'im_file':      im_path,
                        'source':       "{}_{}".format(source, split),
                        'orientation':  get_orientation_chi(im_path.replace("png", "3d"))
                    })


        #### Get Weichao's data
        cameras = ["FusionCameraActor_2", "FusionCameraActor3_4", "FusionCameraActor4_6",
                   "FusionCameraActor5", "FusionCameraActor6", "MonitorCamera"]
        for im_num in tqdm(range(50000), ncols=115, desc="Getting Weichao's data"):
            for cam in cameras:
                scene_file = "/hdd/Datasets/WeichaoCars/scene/{:08d}.json".format(im_num)
                dataset.append({
                    'im_file':      '/hdd/Datasets/WeichaoCars/{}/lit/{:08d}.png'.format(cam, im_num),
                    'source':       'weichao_{}'.format(cam),
                    'orientation':  get_orientation_weichao(scene_file, cam)
                })


        #### Get EPFL data
        seq_info = open("/hdd/Datasets/EPFL/tripod-seq.txt", "r").read().split('\n')
        df = pd.DataFrame({
            'total_frames': [int(x) for x in seq_info[1].split()],
            'rotation_frames': [int(x) for x in seq_info[4].split()],
            'front_frame': [int(x) for x in seq_info[5].split()],
            'rotation_dir': [int(x) for x in seq_info[6].split()],
        })
        for seq_idx, row in tqdm(df.iterrows(), ncols=115, desc="Getting EPFL data", total=len(df)):
            times = open("/hdd/Datasets/EPFL/times_{:02d}.txt".format(seq_idx+1), "r").read().split('\n')
            times = [datetime.datetime.strptime(x, '%Y:%m:%d %H:%M:%S') for x in times[:-1]]
            total_rotation_time = (times[row['rotation_frames'] - 1] - times[0]).total_seconds()
            front_degree_fraction = (times[row['front_frame'] - 1] - times[0]).total_seconds() / total_rotation_time

            for frame in range(0, row['rotation_frames'], 10):
                fraction_through_rotation = (times[frame] - times[0]).total_seconds() / total_rotation_time
                current_orientation = -90 - (-1 * row['rotation_dir'] * (front_degree_fraction - fraction_through_rotation) * 360)

                dataset.append({
                    'im_file':     '/hdd/Datasets/EPFL/tripod_seq_{:02d}_{:03d}.jpg'.format(seq_idx+1, frame+1),
                    'orientation': current_orientation,
                    'source':      'EPFL',
                })


        #### Make sure orientation is in correct coordinate bounds
        df = pd.DataFrame(dataset)
        df.loc[df.orientation > 180, "orientation"] -= 360
        df.loc[df.orientation < -180, "orientation"] += 360


        #### Save to csv
        print(df.sample(25))
        df.to_csv(save_file, index=False)





    def __len__(self):
        if self.split == "train":
            return 50000
        elif self.split == "val":
            return 10000
        else:
            return(len(self.dataset))





    def get_image(self, im_file):
        # Load image
        im = Image.open(im_file).convert('RGB')

        # Transform image
        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ColorJitter(.1,.1,.1,.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transforms_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transforms_train(im) if self.split == "train" else transforms_test(im)





    def convert_orientation_to_class(self, orientation, bins=36):
        return ((np.round(orientation)%360)/(360/bins)).astype(int)





    def __getitem__(self, idx):
        if self.split in ["train", "val"]:
            row = self.dataset.iloc[np.random.randint(len(self.dataset))]
        else:
            row = self.dataset.iloc[idx]

        return {
            'image': self.get_image(row['im_file']),
            'image_file': row['im_file'],
            'source': row['source'],
            'orientation': row['orientation_class'],
        }





    def show(self, idx):
        # TODO: Show the image and the orientation
        pass





def get_angle_between_points(point1, point2):
    # Points are in right-handed coordinate system - get the angle in the x,-z plane
    return np.arctan2(-(point2[2]-point1[2]), (point2[0]-point1[0]))*(180.0/np.pi)





def get_orientation_chi(keypoint_file):
    '''
    Orientation of the car in Chi's data isn't available directly.
    We must get the keypoints and calculate the orientation of the vector between the rear and
        front tire.
    '''
    # Chi's keypoints are x=left-right, y=up-down, z=in-out
    # We want             x=left-right, y=down-up, z=out-in (right-handed coordinate system)
    array = np.array(pd.read_csv(keypoint_file, header=None, sep=" "))
    keypoints = np.array([array[0], -array[1], -array[2]])

    return get_angle_between_points(keypoints[:,17], keypoints[:,16])





def get_orientation_weichao(scene_json_file, camera):
    ''' Orientation of the car in Weichao's data is just the yaw of the camera '''
    loc_rot_rgb_dict = json.load(open(scene_json_file, 'r'))
    return loc_rot_rgb_dict[camera]["Rotation"]["Yaw"]





if __name__ == '__main__':
    ds = CarOrientationDataset("train")
