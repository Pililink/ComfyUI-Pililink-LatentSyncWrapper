# ComfyUI-Pililink-LatentSyncWrapper

这是一个面向 **ComfyUI** 的 LatentSync 1.5 自定义节点封装，重点是：

- 使用 ComfyUI 标准模型目录
- 长视频更稳定（分段推理，降低内存峰值）
- 支持工作流中断

本仓库的实现与改造基于上游 LatentSync ComfyUI 封装项目进行调整。

---

## 主要特性

1. **模型目录规范化**
   - 使用：`ComfyUI/models/latensync1.5/`
   - 兼容旧目录：`latentsync1.5` / `LatentSync-1.5`

2. **长视频防崩策略**
   - 分段推理 + 分段写视频，避免一次性堆积全部帧与中间张量
   - 新增参数：`segment_inferences`（默认 16）

3. **可中断执行**
   - 接入 ComfyUI 中断检查
   - 在关键循环与 ffmpeg 合成阶段都可响应中断

4. **节点命名（Pililink）**
   - 主节点：`Pililink LatentSync 1.5`
   - 路径版节点：`Pililink LatentSync 1.5 (Video Path)`
   - 辅助节点：`Pililink LatentSync Length Adjuster`

---

## 安装

将仓库放到 `ComfyUI/custom_nodes/` 下：

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url>
cd ComfyUI-Pililink-LatentSyncWrapper
pip install -r requirements.txt
```

然后重启 ComfyUI。

---

## 依赖与前置

- 已安装可用的 ComfyUI
- 系统可用 `ffmpeg`（需在 PATH 中）

首次运行会自动检查并下载所需模型文件。

建议同时安装 `DeepCache`，它会在 CUDA 环境下自动启用，用来减少推理阶段的重复 UNet 计算并提升速度。

---

## 模型文件位置

默认模型根目录：

```text
ComfyUI/models/latensync1.5/
```

典型结构示例：

```text
latensync1.5/
├─ latentsync_unet.pt
├─ whisper/
│  └─ tiny.pt
└─ auxiliary/
```

---

## 节点参数说明（主节点）

- `images`：输入图像/帧
- `audio`：输入音频
- `seed`：随机种子
- `lips_expression`：口型表达强度（内部映射 guidance）
- `inference_steps`：推理步数
- `vram_usage`：显存档位（high / medium / low）
- `segment_inferences`：每段推理的 chunk 数（长视频建议 8~24）
- `deepcache`：是否启用 DeepCache（CUDA 环境下推荐保持 `on`）
- `deepcache_cache_interval`：DeepCache 刷新间隔，越大通常越快，但质量风险也更高
- `deepcache_branch_id`：DeepCache 缓存分支，越小通常越激进

> 长视频建议：
> - 先用 `segment_inferences=8~16`
> - `vram_usage=low/medium`

---

## 节点参数说明（路径版节点）

- `video_path`：本地视频绝对路径
- `audio`：可选，ComfyUI 音频输入
- `audio_path`：可选，本地音频绝对路径；填写后优先于 `audio`
- `deepcache`：是否启用 DeepCache（CUDA 环境下推荐保持 `on`）
- `deepcache_cache_interval`：DeepCache 刷新间隔，默认 `3`
- `deepcache_branch_id`：DeepCache 缓存分支，默认 `0`
- `clip_batch_size`：每次送入 refactor 运行时的 clip 批大小，显存紧张时可调小
- `auto_oom_fallback`：显存不足时自动回退到更小的 clip 批大小
- `quality_mode`：`balanced` / `quality_first`，后者更保守、显存和速度通常更高但更稳
- `mode`：时长对齐模式，支持 `normal` / `pingpong` / `loop_to_audio`
- `silent_padding_sec`：补齐音频时附加的静音秒数
- `auto_silent_padding`：自动根据视频和音频时长差计算静音补齐时长；例如视频 50s、音频 10s 时，会自动补约 40s 静音
- `result_mode`：结果输出模式，`memory_only` 表示仅把结果交给后续节点并自动清理过程文件，`both` 表示同时保留输出 mp4 文件
- `filename_prefix`：输出到 ComfyUI `output` 目录时使用的前缀
- `output_path`：可选，直接指定最终输出 mp4 的绝对路径
- `seed` / `lips_expression` / `inference_steps` / `vram_usage` / `segment_inferences`：与主节点一致

路径版节点会直接读取源视频路径并输出保存后的 mp4 路径和音频，不返回 `IMAGE` tensor，因此不受视频长度限制，适合长视频处理。`result_mode` 默认 `both`，输出文件会保存到 ComfyUI 的 `output` 目录。

如果 `audio` 和 `audio_path` 都留空，节点会自动尝试提取输入视频自带音轨作为驱动音频。

时长模式说明：

- `normal`：对齐到较短一侧；如果视频比音频长，会先给音频补一小段静音再裁视频；如果音频比视频长，则裁音频
- `pingpong`：当音频比视频长时，把视频做往返播放扩展到音频时长；当音频较短时，保留原视频并给音频补静音
- `loop_to_audio`：循环视频直到覆盖音频时长，并可附加静音尾巴

---

## 长视频推荐工作流

长视频尽量不要走 `Load Video (Upload) 🎥🅥🅗🅢 -> IMAGE` 这条链路，因为它会在进入 Pililink 之前先把整段视频展开到内存。

推荐改成：

```text
字符串路径 / 路径输入节点
        -> Pililink LatentSync 1.5 (Video Path)
        -> 输出 mp4 路径
```

如果还需要单独加载音频，可继续使用音频节点接到 `audio` 输入；如果视频原音就是驱动音频，可以直接只填 `video_path`。

---

## 输出与行为

- 输出：处理后视频帧 + 音频（ComfyUI 标准输出）
- 路径版输出：返回 `audio`、`video_path`、`filename` 3 个输出，方便继续走文件路径链路
- 兼容性：旧的 `Pililink LatentSync 1.5 (Video Path, Refactor Legacy)` 仍保留为兼容别名，推荐新建工作流时直接使用 `Pililink LatentSync 1.5 (Video Path)`
- 当 `result_mode=memory_only` 且未填写 `output_path` 时，结果文件只会暂存于临时目录，读取完成后自动清理；这时 `video_path` 和 `filename` 输出会为空字符串
- 执行中可在 ComfyUI 中断（Stop）
- 临时文件在节点运行目录下自动管理

---

## 已知说明

- 输入视频会按流程要求处理为 25fps 路径后再推理
- 对超长视频，速度会下降但稳定性更高（这是分段策略的预期）
- 如果已安装 `DeepCache`，当前版本会自动启用；如需临时排查兼容问题，可设置环境变量 `PILILINK_DISABLE_DEEPCACHE=1` 后再启动 ComfyUI
- 推荐起步组合：`deepcache=on`、`deepcache_cache_interval=3`、`deepcache_branch_id=0`
- 如果你更追求速度，可以试 `deepcache_cache_interval=5`；如果画质或口型稳定性下降，再退回 `3`

---

## 致谢

- [GeekyGhost / ComfyUI-Geeky-LatentSyncWrapper](https://github.com/GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper)
- [ByteDance LatentSync](https://github.com/bytedance/LatentSync)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
