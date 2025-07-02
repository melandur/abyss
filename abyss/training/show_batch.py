import matplotlib.pyplot as plt
from create_dataset import get_loader

from config import ConfigFile

config_file = ConfigFile()
config = config_file.get_config()
config['training']['batch_size'] = 1

train_loader = get_loader(config, 'train')
out_channels = len(config['trainer']['label_classes'])

plt.ion()
plt.figure("image", (12, 7))


for idx, batch in enumerate(train_loader):
    print(f"image shape: {batch['image'].shape}")

    in_channels = batch["image"].shape[1]
    out_channels = batch["label"].shape[1]

    for i in range(in_channels):
        plt.subplot(4, 4, i + 1)
        # plt.title(f"image channel {i}")
        middle = batch["image"].shape[-1] // 2
        plt.imshow(batch["image"][0, i, :, :, middle].detach().cpu(), cmap="gray")

    print(f"image shape: {batch['label'].shape}")
    # plt.figure("label", (18, 6))
    for i in range(out_channels):
        ix = i + 4
        plt.subplot(4, 4, ix + 1)
        # plt.title(f"label channel {i}")
        middle = batch["label"].shape[-1] // 2
        plt.imshow(batch["label"][0, i, :, :, middle].detach().cpu(), filternorm=False)

    for i in range(in_channels):
        ix = i + 8
        plt.subplot(4, 4, ix + 1)
        # plt.title(f"image channel {i}")
        middle = batch["image"].shape[-1] // 2
        plt.imshow(batch["image"][1, i, :, :, middle].detach().cpu(), cmap="gray")

    # plt.figure("label", (18, 6))
    for i in range(out_channels):
        ix = i + 12
        plt.subplot(4, 4, ix + 1)
        # plt.title(f"label channel {i}")
        middle = batch["label"].shape[-1] // 2
        plt.imshow(batch["label"][1, i, :, :, middle].detach().cpu(), filternorm=False)

    plt.draw()
    plt.pause(0.0001)
    plt.waitforbuttonpress()
    plt.clf()


# val_loader = get_loader(config, 'val')
# for idx, batch in enumerate(val_loader):
#     print(f"image shape: {batch['image'].shape}")
#
#     for i in range(4):
#         plt.subplot(3, 4, i + 1)
#         # plt.title(f"image channel {i}")
#         middle = batch["image"].shape[-1] // 2
#         plt.imshow(batch["image"][0, i, :, :, middle].detach().cpu(), cmap="gray")
#
#     print(f"image shape: {batch['label'].shape}")
#     # plt.figure("label", (18, 6))
#     for i in range(2):
#         ix = i + 4
#         plt.subplot(3, 4, ix + 1)
#         # plt.title(f"label channel {i}")
#         middle = batch["label"].shape[-1] // 2
#         plt.imshow(batch["label"][0, i, :, :, middle].detach().cpu(), filternorm=False)
#
#     plt.draw()
#     plt.pause(0.0001)
#     plt.waitforbuttonpress()
#     plt.clf()
