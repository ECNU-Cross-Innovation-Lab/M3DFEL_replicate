import os
from torch.utils import data
import torch
import glob
import os
import numpy as np
import csv
import PIL.Image as Image
import torchvision

from .video_transform import *

class DFEWDataset(data.Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.path = self.args.train_dataset if mode == "train" else self.args.test_dataset
        self.num_frames = self.args.num_frames
        self.image_size = self.args.crop_size
        self.mode = mode
        self.transform = self.get_transform()
        self.data = self.get_data()
            
        pass

    def get_data(self):
        full_data = []

        npy_path = self.path.replace('csv', 'npy')
        print("loading data")
        if os.path.exists(npy_path):
            full_data = np.load(npy_path, allow_pickle=True)
        else:
            with open(self.path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    path = row[0]
                    emotion = int(row[1]) - 1
                    while len(path) < 5:
                        path = "0" + path
                    path = os.path.join(self.args.root, "Clip/clip_224x224/clip_224x224/", path)
                    full_num_frames = len(os.listdir(path))

                    full_video_frames_paths = glob.glob(os.path.join(path, '*.jpg'))
                    full_video_frames_paths.sort()

                    full_data.append({"path": full_video_frames_paths, "emotion": emotion, "num_frames": full_num_frames})
        
                np.save(npy_path, full_data)
        
        print("data loaded")
        return full_data

    def get_transform(self):

        transform = None
        if self.mode == "train":
            transform = torchvision.transforms.Compose([GroupRandomSizedCrop(self.image_size),
                                                        GroupRandomHorizontalFlip(),
                                                        GroupColorJitter(self.args.color_jitter),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        elif self.mode == "test":
            transform = torchvision.transforms.Compose([GroupResize(self.image_size),
                                                        Stack(),
                                                        ToTorchFormatTensor()])
        
        return transform

    def __getitem__(self, index):
        data = self.data[index]

        full_video_frames_paths = data['path']

        video_frames_paths = []
        full_num_frames = len(full_video_frames_paths)
        for i in range(self.num_frames):

            frame = int(full_num_frames * i / self.num_frames)
            if self.args.random_sample:
                frame += int(random.random() * self.num_frames)
                frame = min(full_num_frames - 1, frame)
            video_frames_paths.append(full_video_frames_paths[frame])

        images = []
        for video_frames_path in video_frames_paths:
            images.append(Image.open(video_frames_path).convert('RGB'))

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        return images, data["emotion"]

    def __len__(self):
        return len(self.data)



