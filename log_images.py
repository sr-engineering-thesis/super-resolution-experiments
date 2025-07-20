import torch
from constants import TEST_LR_DATA_PATH, TEST_HR_DATA_PATH, RESULTS_DIR_PATH
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor

IMAGES_INDICES = [949, 1819]


def log_images(model, epoch, config, hash_for_run):
    for idx in IMAGES_INDICES:
        test_image = cv2.imread(str(TEST_LR_DATA_PATH / f"{idx}.png"), cv2.IMREAD_COLOR)
        test_image_ycrcb = cv2.cvtColor(test_image, cv2.COLOR_BGR2YCrCb)
        backup_dims = test_image_ycrcb[:, :, 1:]
        test_image = test_image_ycrcb[:, :, 0]
        test_image = to_tensor(test_image).to(config.device)

        with torch.no_grad():
            test_image = test_image.unsqueeze(0)
            sr_test_image = model(test_image).squeeze().cpu().numpy()

        sr_test_image_uint8 = np.clip(sr_test_image * 255.0, 0, 255).astype(np.uint8)

        height, width = sr_test_image_uint8.shape
        cr_resized = cv2.resize(
            backup_dims[:, :, 0], (width, height), interpolation=cv2.INTER_CUBIC
        )
        cb_resized = cv2.resize(
            backup_dims[:, :, 1], (width, height), interpolation=cv2.INTER_CUBIC
        )

        ycrcb_merged = cv2.merge([sr_test_image_uint8, cr_resized, cb_resized])
        sr_bgr = cv2.cvtColor(ycrcb_merged, cv2.COLOR_YCrCb2BGR)

        # sr image saving
        cv2.imwrite(
            f"{RESULTS_DIR_PATH}/{idx}_{hash_for_run}_sr_{epoch}.png",
            sr_bgr,
        )

        # lr upscaled image saving
        test_image_np = test_image.cpu().numpy().squeeze()
        test_image_resized = cv2.resize(
            test_image_np, (width, height), interpolation=cv2.INTER_CUBIC
        )
        test_image_uint8 = np.clip(test_image_resized * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(
            f"{RESULTS_DIR_PATH}/{idx}_{hash_for_run}_lr_{epoch}.png",
            test_image_uint8,
        )

        # hr image saving
        hr_image = cv2.imread(str(TEST_HR_DATA_PATH / f"{idx}.png"), cv2.IMREAD_COLOR)
        cv2.imwrite(
            f"{RESULTS_DIR_PATH}/{idx}_{hash_for_run}_hr_{epoch}.png",
            hr_image,
        )
