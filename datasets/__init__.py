from .dataset_DFEW import DFEWDataset
from .dataset_FERV39K import FERV39KDataset
import torch

def create_dataloader(args, mode):

    if args.dataset == 'DFEW':
        dataset = DFEWDataset(args, mode)
    elif args.dataset == 'FERV39K':
        dataset = FERV39KDataset(args, mode)

    dataloader = None
    if mode == "train":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True,
                                                drop_last=True)
    elif mode == "test":
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.workers,
                                                pin_memory=True, 
                                                drop_last=True)
    return dataloader
