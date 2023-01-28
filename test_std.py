"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import h5py
import pdb
import numpy
import os
import sys
import torch

sys.path.append(os.path.join("..", "util"))
import common_pelvic_pt as common_pelvic

def main():
    device = torch.device("cpu")
    
    data_iter = common_pelvic.DataIter(device, r"F:\datasets\pelvic\h5_data_nonrigid", patch_depth=3, batch_size=1)
    
    for i in range(10000):
        patch_s, patch_t, _ = data_iter.next()
        
        if i % 1000 == 0:
            print(i)
        try:
            assert patch_s.std() > 0 and patch_t.std() > 0
        except:
            print("err", i)
        
    print("Done")


if __name__ == '__main__':
    main()

