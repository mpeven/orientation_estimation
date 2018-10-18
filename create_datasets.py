import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
pd.set_option('display.width', 200)


def chi():
    '''
    Get Chi's data
    '''

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


    dataset = []
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

    return dataset





def weichao():
    '''
    Get Weichao's data
    '''

    def get_orientation_weichao(scene_json_file, camera):
        ''' Orientation of the car in Weichao's data is just the yaw of the camera '''
        loc_rot_rgb_dict = json.load(open(scene_json_file, 'r'))
        return loc_rot_rgb_dict[camera]["Rotation"]["Yaw"]


    dataset = []
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

    return dataset





def EPFL():
    '''
    Get EPFL data
    '''

    dataset = []
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

        for frame in range(0, row['rotation_frames']):
            fraction_through_rotation = (times[frame] - times[0]).total_seconds() / total_rotation_time
            current_orientation = -90 - (-1 * row['rotation_dir'] * (front_degree_fraction - fraction_through_rotation) * 360)

            dataset.append({
                'im_file':     '/hdd/Datasets/EPFL/tripod_seq_{:02d}_{:03d}.jpg'.format(seq_idx+1, frame+1),
                'source':      'EPFL',
                'orientation': current_orientation,
            })

    return dataset





def objectNet3D():
    '''
    Get ObjectNet3D data & save out cropped images
    '''

    dataset = []
    import scipy.io as sio

    for matfile in glob.glob("/Users/mpeven/Downloads/ObjectNet3D/Annotations/*")[:200]:
        x = sio.loadmat(matfile)
        image = x['record']['filename'][0,0][0]
        for obj in x['record']['objects'][0,0][0]:
            if obj['class'][0] != 'car' or 'azimuth_coarse' not in obj['viewpoint'].dtype.names:
                continue
            print("Filename:", image)
            print("Class:", obj['class'])
            print("Bbox: ", obj['bbox'][0])

            print("Options: ", obj['viewpoint'].dtype.names)
            if 'azimuth' in obj['viewpoint'].dtype.names:
                print("Azimuth:", obj['viewpoint']['azimuth'])
            if 'azimuth_coarse' in obj['viewpoint'].dtype.names:
                print("Azimuth_coarse:", obj['viewpoint']['azimuth_coarse'])
            print("\n\n")





def WCVP():
    '''
    Get Weizmann Cars ViewPoint data
    '''

    dataset = []
    for ann_file in tqdm(glob.glob("/hdd/Datasets/WCVP/*/*.txt"), ncols=115, desc="Getting WCVP data"):
        xyz = open(ann_file, "r").readlines()[3].split()
        dataset.append({
            'im_file': ann_file.replace('txt', 'jpg'),
            'orientation': np.arctan2(float(xyz[0]), float(xyz[1])) * 180. / np.pi,
            'source': 'WCVP',
        })

    return dataset




def DIVA():
    '''
    Get DIVA tracks
    '''
    dataset = []
    df = pd.read_csv("/home/mike/Projects/DIVA/geometric_methods/datasets/dataset_yaml.csv")
    for track_id, track_df in df.groupby("track_id"):
        # if track_df.iloc[0]["video"] != "VIRAT_S_040003_04_000758_001118.mp4":
        #     continue
        # print(track_id, track_df)
        # continue
        if track_id != 2195:
            continue

        for _, row in track_df.iloc[-700:].iterrows():
            dataset.append({
                'im_file': "/hdd/Datasets/DIVA/diva_data/{}/{}/{:05d}.png".format(row['video'][8:12], row['video'].replace(".mp4", ""), row['frame']),
                'orientation': 0,
                'source': "DIVA",
                'bbox': "{} {} {} {}".format(row['xmin'], row['ymin'], row['xmax'], row['ymax']),
            })

    return dataset




def main():
    #### Get the datasets
    datasets = [
        # pd.read_csv("car_orientation_dataset.csv"),
        # chi(),
        # weichao(),
        # EPFL(),
        # WCVP(),
        DIVA(),
    ]

    #### Combine into dataframe
    df = pd.concat([pd.DataFrame(ds) for ds in datasets]).reset_index(drop=True)

    #### Make sure orientation is in correct coordinate bounds
    df.loc[df.orientation > 180, "orientation"] -= 360
    df.loc[df.orientation < -180, "orientation"] += 360


    #### Save to csv
    print(df.sample(25))
    df.to_csv("car_orientation_dataset_diva.csv", index=False)





if __name__ == '__main__':
    main()
