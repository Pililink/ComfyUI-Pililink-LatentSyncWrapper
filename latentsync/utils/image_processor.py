# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import transforms
import cv2
from einops import rearrange
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Union
from .affine_transform import AlignRestore, laplacianSmooth, transformation_from_points
import face_alignment

"""
If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image, you should prefer to use INTER_AREA interpolation.
https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
"""


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image).to(dtype=torch.float32), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, mask: str = "fix_mask", device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.device_str = "cuda" if str(device).lower() != "cpu" and torch.cuda.is_available() else "cpu"
        self.torch_device = torch.device(self.device_str)
        self.use_gpu_affine = self.device_str == "cuda"
        detect_size_env = os.environ.get("LATENTSYNC_FACE_DETECT_MAX_SIZE", "960")
        self.max_detection_size = max(256, int(detect_size_env))
        self._output_coord_cache = {}
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.mask = mask

        if mask in ["mouth", "face", "eye"]:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
        if mask == "fix_mask":
            self.face_mesh = None
            self.smoother = laplacianSmooth()
            self.restorer = AlignRestore()

            if mask_image is None:
                self.mask_image = load_fixed_mask(resolution)
            else:
                self.mask_image = mask_image.to(dtype=torch.float32)

            self.mask_image = self.mask_image.to(device=self.torch_device, dtype=torch.float32)

            if self.device_str != "cpu":
                self.fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D, flip_input=False, device=self.device_str
                )
                self.face_mesh = None
            else:
                # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)  # Process single image
                self.face_mesh = None
                self.fa = None

    def _get_warp_target_coords(self, out_h: int, out_w: int):
        cache_key = (out_h, out_w, str(self.torch_device))
        cached = self._output_coord_cache.get(cache_key)
        if cached is not None:
            return cached

        ys = torch.arange(out_h, dtype=torch.float32, device=self.torch_device)
        xs = torch.arange(out_w, dtype=torch.float32, device=self.torch_device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        ones = torch.ones_like(xx)
        coords = torch.stack([xx, yy, ones], dim=-1).reshape(-1, 3).transpose(0, 1).contiguous()
        self._output_coord_cache[cache_key] = coords
        return coords

    def _detect_landmarks(self, image: np.ndarray, allow_multi_faces: bool = True):
        if self.fa is None:
            landmark_coordinates = np.array(self.detect_facial_landmarks(image))
            return mediapipe_lm478_to_face_alignment_lm68(landmark_coordinates)

        detect_image = image
        detect_scale = 1.0
        height, width = image.shape[:2]
        max_dim = max(height, width)
        if self.max_detection_size > 0 and max_dim > self.max_detection_size:
            detect_scale = self.max_detection_size / float(max_dim)
            detect_image = cv2.resize(
                image,
                (
                    max(2, int(round(width * detect_scale))),
                    max(2, int(round(height * detect_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )

        detected_faces = self.fa.get_landmarks(detect_image)
        if detected_faces is None:
            raise RuntimeError("Face not detected")
        if not allow_multi_faces and len(detected_faces) > 1:
            raise RuntimeError("More than one face detected")

        lm68 = np.asarray(detected_faces[0], dtype=np.float32)
        if detect_scale != 1.0:
            lm68 = lm68 / detect_scale
        return lm68

    def _warp_face_cpu(self, image: np.ndarray, affine_matrix: np.ndarray):
        face = cv2.warpAffine(
            image,
            affine_matrix,
            self.restorer.face_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=[127, 127, 127],
        )
        if face.shape[0] != self.resolution or face.shape[1] != self.resolution:
            face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(np.ascontiguousarray(face)), "h w c -> c h w")
        return face

    def _warp_face_gpu(self, image: np.ndarray, affine_matrix: np.ndarray):
        out_w, out_h = self.restorer.face_size
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(device=self.torch_device, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        src_h, src_w = image.shape[:2]
        coords = self._get_warp_target_coords(out_h, out_w)

        affine = torch.tensor(affine_matrix, device=self.torch_device, dtype=torch.float32)
        affine3 = torch.eye(3, device=self.torch_device, dtype=torch.float32)
        affine3[:2, :] = affine
        inv_affine = torch.linalg.inv(affine3)[:2, :]
        src_coords = inv_affine @ coords
        src_x = src_coords[0].reshape(out_h, out_w)
        src_y = src_coords[1].reshape(out_h, out_w)

        if src_w > 1:
            grid_x = (src_x / (src_w - 1.0)) * 2.0 - 1.0
        else:
            grid_x = torch.zeros_like(src_x)
        if src_h > 1:
            grid_y = (src_y / (src_h - 1.0)) * 2.0 - 1.0
        else:
            grid_y = torch.zeros_like(src_y)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        centered = image_tensor - 127.0
        sampled = F.grid_sample(
            centered,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ) + 127.0

        if out_h != self.resolution or out_w != self.resolution:
            sampled = F.interpolate(
                sampled,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            )

        face = sampled.squeeze(0).clamp(0.0, 255.0).round().to(torch.uint8)
        return face

    def warp_face_with_affine_matrix(self, image: np.ndarray, affine_matrix: np.ndarray):
        if self.use_gpu_affine:
            face = self._warp_face_gpu(image, affine_matrix)
        else:
            face = self._warp_face_cpu(image, affine_matrix)
        box = [0, 0, int(self.restorer.face_size[0]), int(self.restorer.face_size[1])]
        return face, box

    def detect_facial_landmarks(self, image: np.ndarray):
        height, width, _ = image.shape
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:  # Face not detected
            raise RuntimeError("Face not detected")
        face_landmarks = results.multi_face_landmarks[0]  # Only use the first face in the image
        landmark_coordinates = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark
        ]  # x means width, y means height
        return landmark_coordinates

    def preprocess_one_masked_image(self, image: torch.Tensor) -> np.ndarray:
        image = self.resize(image)

        if self.mask == "mouth" or self.mask == "face":
            landmark_coordinates = self.detect_facial_landmarks(image)
            if self.mask == "mouth":
                surround_landmarks = mouth_surround_landmarks
            else:
                surround_landmarks = face_surround_landmarks

            points = [landmark_coordinates[landmark] for landmark in surround_landmarks]
            points = np.array(points)
            mask = np.ones((self.resolution, self.resolution))
            mask = cv2.fillPoly(mask, pts=[points], color=(0, 0, 0))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        elif self.mask == "half":
            mask = torch.ones((self.resolution, self.resolution))
            height = mask.shape[0]
            mask[height // 2 :, :] = 0
            mask = mask.unsqueeze(0)
        elif self.mask == "eye":
            mask = torch.ones((self.resolution, self.resolution))
            landmark_coordinates = self.detect_facial_landmarks(image)
            y = landmark_coordinates[195][1]
            mask[y:, :] = 0
            mask = mask.unsqueeze(0)
        else:
            raise ValueError("Invalid mask type")

        image = image.to(dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * mask
        mask = 1 - mask

        return pixel_values, masked_pixel_values, mask

    def affine_transform(self, image: torch.Tensor, allow_multi_faces: bool = True) -> np.ndarray:
        lm68 = self._detect_landmarks(image, allow_multi_faces=allow_multi_faces)

        points = self.smoother.smooth(lm68)
        lmk3_ = np.zeros((3, 2))
        lmk3_[0] = points[17:22].mean(0)
        lmk3_[1] = points[22:27].mean(0)
        lmk3_[2] = points[27:36].mean(0)
        affine_matrix, self.restorer.p_bias = transformation_from_points(
            lmk3_,
            self.restorer.face_template,
            smooth=True,
            p_bias=self.restorer.p_bias,
        )
        face, box = self.warp_face_with_affine_matrix(image, affine_matrix)
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        image = image.to(device=self.torch_device, dtype=torch.float32)
        pixel_values = self.normalize(image / 255.0)
        mask_image = self.mask_image.to(device=pixel_values.device, dtype=pixel_values.dtype)
        masked_pixel_values = pixel_values * mask_image
        return pixel_values, masked_pixel_values, mask_image[0:1]

    @staticmethod
    def _ensure_nchw_tensor(images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() == 4 and images.shape[-1] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        return images

    def _prepare_fixed_mask_batch(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        images = self._ensure_nchw_tensor(images)

        if affine_transform:
            results = [self.preprocess_fixed_mask_image(image, affine_transform=True) for image in images]
            pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
            return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

        images = images.to(device=self.torch_device, dtype=torch.float32)
        images = self.resize(images)
        pixel_values = (images / 255.0 - 0.5) / 0.5
        mask_image = self.mask_image.to(device=pixel_values.device, dtype=pixel_values.dtype)
        masked_pixel_values = pixel_values * mask_image.unsqueeze(0)
        masks = mask_image[0:1].unsqueeze(0).expand(pixel_values.shape[0], -1, -1, -1)
        return pixel_values, masked_pixel_values, masks

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        images = self._ensure_nchw_tensor(images)
        if self.mask == "fix_mask":
            return self._prepare_fixed_mask_batch(images, affine_transform=affine_transform)
        else:
            results = [self.preprocess_one_masked_image(image) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()


def mediapipe_lm478_to_face_alignment_lm68(lm478, return_2d=True):
    """
    lm478: [B, 478, 3] or [478,3]
    """
    # lm478[..., 0] *= W
    # lm478[..., 1] *= H
    landmarks_extracted = []
    for index in landmark_points_68:
        x = lm478[index][0]
        y = lm478[index][1]
        landmarks_extracted.append((x, y))
    return np.array(landmarks_extracted)


landmark_points_68 = [
    162,
    234,
    93,
    58,
    172,
    136,
    149,
    148,
    152,
    377,
    378,
    365,
    397,
    288,
    323,
    454,
    389,
    71,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    301,
    168,
    197,
    5,
    4,
    75,
    97,
    2,
    326,
    305,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    39,
    37,
    0,
    267,
    269,
    291,
    405,
    314,
    17,
    84,
    181,
    78,
    82,
    13,
    312,
    308,
    317,
    14,
    87,
]


# Refer to https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
mouth_surround_landmarks = [
    164,
    165,
    167,
    92,
    186,
    57,
    43,
    106,
    182,
    83,
    18,
    313,
    406,
    335,
    273,
    287,
    410,
    322,
    391,
    393,
]

face_surround_landmarks = [
    152,
    377,
    400,
    378,
    379,
    365,
    397,
    288,
    435,
    433,
    411,
    425,
    423,
    327,
    326,
    94,
    97,
    98,
    203,
    205,
    187,
    213,
    215,
    58,
    172,
    136,
    150,
    149,
    176,
    148,
]

if __name__ == "__main__":
    image_processor = ImageProcessor(512, mask="fix_mask")
    video = cv2.VideoCapture("assets/demo1_video.mp4")
    while True:
        ret, frame = video.read()
        # if not ret:
        #     break

        # cv2.imwrite("image.jpg", frame)

        frame = rearrange(torch.Tensor(frame).type(torch.uint8), "h w c ->  c h w")
        # face, masked_face, _ = image_processor.preprocess_fixed_mask_image(frame, affine_transform=True)
        face, _, _ = image_processor.affine_transform(frame)

        break

    face = (rearrange(face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    cv2.imwrite("face.jpg", face)

    # masked_face = (rearrange(masked_face, "c h w -> h w c").detach().cpu().numpy()).astype(np.uint8)
    # cv2.imwrite("masked_face.jpg", masked_face)
