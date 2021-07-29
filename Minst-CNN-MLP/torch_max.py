"""
 Env: /anaconda3/python3.7
 Time: 2021/7/14 20:10
 Author: karlieswfit
 File: torch_max.py
 Describe: 
"""

import torch
import numpy as np
data=torch.tensor(np.arange(12).reshape(3,4))
print(data)
print(data.max(dim=-1))
# torch.return_types.max(
# values=tensor([ 3,  7, 11], dtype=torch.int32),
# indices=tensor([3, 3, 3]))
