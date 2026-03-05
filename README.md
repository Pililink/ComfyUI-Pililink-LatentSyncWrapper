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

> 长视频建议：
> - 先用 `segment_inferences=8~16`
> - `vram_usage=low/medium`

---

## 输出与行为

- 输出：处理后视频帧 + 音频（ComfyUI 标准输出）
- 执行中可在 ComfyUI 中断（Stop）
- 临时文件在节点运行目录下自动管理

---

## 已知说明

- 输入视频会按流程要求处理为 25fps 路径后再推理
- 对超长视频，速度会下降但稳定性更高（这是分段策略的预期）

---

## 致谢

- [GeekyGhost / ComfyUI-Geeky-LatentSyncWrapper](https://github.com/GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper)
- [ByteDance LatentSync](https://github.com/bytedance/LatentSync)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
