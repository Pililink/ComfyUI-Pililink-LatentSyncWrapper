import time
import os
import numpy as np
import cv2
import torch
from .nets import S3FDNet
from .box_utils import nms_


def _candidate_latentsync_roots():
    candidates = []

    env_root = os.environ.get("LATENTSYNC_ROOT_DIR")
    if env_root:
        candidates.append(env_root)

    try:
        folder_paths_mod = __import__("folder_paths")
        comfy_models = folder_paths_mod.models_dir
    except Exception:
        comfy_models = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "models",
        )

    candidates.append(os.path.join(comfy_models, "latensync1.5"))
    candidates.append(os.path.join(comfy_models, "latentsync1.5"))
    candidates.append(os.path.join(comfy_models, "LatentSync-1.5"))

    candidates.append("checkpoints")

    seen = set()
    unique = []
    for path in candidates:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    return unique


def _resolve_s3fd_weight_path() -> str:
    for root in _candidate_latentsync_roots():
        path = os.path.join(root, "auxiliary", "sfd_face.pth")
        if os.path.exists(path):
            return path

    return os.path.join(_candidate_latentsync_roots()[0], "auxiliary", "sfd_face.pth")


PATH_WEIGHT = _resolve_s3fd_weight_path()
img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:

    def __init__(self, device="cuda"):

        tstamp = time.time()
        self.device = device

        print("[S3FD] loading with", self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print("[S3FD] finished loading (%.4f sec)" % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype("float32")
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes
