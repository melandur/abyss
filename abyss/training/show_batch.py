# pick one image from DecathlonDataset to visualize and check the 4 channels
import matplotlib.pyplot as plt
from create_dataset import get_loader

from abyss.config import ConfigFile

config_file = ConfigFile()
config = config_file.get_config()
config['training']['batch_size'] = 1

train_loader = get_loader(config, 'train')
out_channels = len(config['trainer']['label_classes'])

plt.ion()
plt.figure("image", (12, 7))

for idx, batch in enumerate(train_loader):
    print(f"image shape: {batch['image'].shape}")

    for i in range(4):
        plt.subplot(3, 4, i + 1)
        # plt.title(f"image channel {i}")
        plt.imshow(batch["image"][0, i, :, :, 60].detach().cpu(), cmap="gray")

    print(f"image shape: {batch['label'].shape}")
    # plt.figure("label", (18, 6))
    for i in range(4):
        ix = i + 4
        plt.subplot(3, 4, ix + 1)
        # plt.title(f"label channel {i}")
        plt.imshow(batch["label"][0, i, :, :, 60].detach().cpu(), interpolation=None)

    plt.draw()
    plt.pause(0.0001)
    plt.waitforbuttonpress()
    plt.clf()

    # plt.waitforbuttonpress()
    # plt.close()
