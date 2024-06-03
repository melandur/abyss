import torchio as tio
import monai.transforms as tf


tio.RandomGamma(p=0.5, log_gamma=(0.9, 1.1), keys=['image']),