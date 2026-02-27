# Mobile-O Android

Android 端推理应用，使用 ONNX Runtime 在设备上运行 Mobile-O 多模态模型，支持文本对话、图像理解、图像生成三种模式。

---

## 功能

| 输入 | 行为 |
|------|------|
| `generate <prompt>` | 文本生成图像 |
| 文本 + 附图（无前缀）| 图像理解，返回文本描述 |
| 纯文本 | 普通文本对话 |

---

## 架构概览

```
android/app/src/main/kotlin/com/mobileo/
├── ml/
│   ├── OnnxSessionManager.kt          # 加载和持有全部 5 个 OrtSession
│   ├── Tokenizer.kt                   # Qwen2 BPE 分词器（纯 Kotlin）
│   ├── MobileConditioningConnector.kt # LLM 隐层 → DiT 条件向量
│   ├── MobileOGenerator.kt            # 文本→图像完整管线
│   ├── DPMSolverScheduler.kt          # DPM-Solver++ 调度器（纯 Kotlin）
│   └── InferenceModels.kt             # 图像理解 + 文本对话
├── service/
│   └── ModelDownloadManager.kt        # OkHttp 断点续传下载管理
├── ui/screen/
│   ├── ConversationScreen.kt          # 主聊天界面
│   ├── DownloadGateScreen.kt          # 首次启动下载界面
│   ├── SettingsScreen.kt              # 生成参数设置
│   └── CameraScreen.kt                # 实时摄像头理解
└── viewmodel/
    ├── ChatViewModel.kt               # 主 ViewModel，模型加载与路由
    └── SettingsViewModel.kt           # 生成参数（步数、引导系数等）
```

**运行时依赖：**
- ONNX Runtime Android 1.20.0（含 NNAPI 代理，自动 CPU 兜底）
- NDK NEON 内核用于 VAE 后处理（`vae_postprocess.cpp`）

---

## 设备要求

| 项目 | 要求 |
|------|------|
| Android 版本 | Android 10+（minSdk 29）|
| CPU 架构 | arm64-v8a 或 x86_64 |
| 内存 | 建议 8 GB RAM 以上 |
| 存储空间 | 约 6 GB 可用空间 |

---

## 快速开始（推荐：自动下载）

App 首次启动会自动从 HuggingFace 下载全部模型，无需手动操作。

### 构建与运行

```bash
cd android
./gradlew installDebug
```

首次启动：
1. 出现下载界面，显示所需空间（约 5 GB）
2. 点击 **"Download Models"** 开始下载，支持后台运行和断点续传
3. 下载完成后进入对话界面，之后完全离线运行

模型下载自 [`Amshaker/Mobile-O-0.5B-Android`](https://huggingface.co/Amshaker/Mobile-O-0.5B-Android)，存储在 `getExternalFilesDir(null)/models/`（即 `/sdcard/Android/data/com.mobileo/files/models/`），外部存储不可用时自动回退到内部 `filesDir`。

---

## 手动部署（开发调试）

如果你已经用 `export_onnx.py` 在本地完成了模型导出，可以通过 `adb push` 直接推送模型，跳过下载流程。

### 导出产物说明

`export_onnx.py` 导出完成后会自动处理所有 Android 兼容性问题，产物可直接使用：

```
onnx_models/
├── llm.onnx              ← 图结构（~1.2 MB）
├── llm.onnx.data         ← 全部权重，连续存储带字节偏移（~2.4 GB）
├── transformer.onnx      ← 图结构（~2.1 MB）
├── transformer.onnx.data ← 全部权重（~2.3 GB）
├── vae_decoder.onnx      ← 单文件（~608 MB）
├── connector.onnx        ← 单文件（~9 MB）
├── vision_encoder.onnx   ← 单文件（~468 MB）
└── llm/                  ← Tokenizer 文件
```

LLM 和 DiT 因超过 ONNX 2 GB 协议体积限制，权重存储在 `.onnx.data` sidecar 文件中。Android ORT 加载 `.onnx` 时会自动在同目录下查找对应的 `.onnx.data`。

### 第一步：安装 APK

```bash
cd android
./gradlew installDebug
```

### 第二步：推送模型文件

模型目标路径为 ExternalFilesDir，**无需 root 权限**即可通过 `adb push` 写入：

```bash
# 创建目录
adb shell mkdir -p /sdcard/Android/data/com.mobileo/files/models/llm

# LLM（graph + sidecar 权重）
adb push onnx_models/llm.onnx       /sdcard/Android/data/com.mobileo/files/models/llm.onnx
adb push onnx_models/llm.onnx.data  /sdcard/Android/data/com.mobileo/files/models/llm.onnx.data

# DiT（graph + sidecar 权重）
adb push onnx_models/transformer.onnx       /sdcard/Android/data/com.mobileo/files/models/transformer.onnx
adb push onnx_models/transformer.onnx.data  /sdcard/Android/data/com.mobileo/files/models/transformer.onnx.data

# 单文件模型
adb push onnx_models/connector.onnx      /sdcard/Android/data/com.mobileo/files/models/connector.onnx
adb push onnx_models/vae_decoder.onnx    /sdcard/Android/data/com.mobileo/files/models/vae_decoder.onnx
adb push onnx_models/vision_encoder.onnx /sdcard/Android/data/com.mobileo/files/models/vision_encoder.onnx

# Tokenizer 文件
for f in onnx_models/llm/*; do
    adb push "$f" /sdcard/Android/data/com.mobileo/files/models/llm/
done
```

### 第三步：启动 App

模型文件已存在于目标路径，App 启动时会自动跳过下载界面，直接加载模型。

---

## 模型导出

模型由项目根目录的 `export_onnx.py` 生成，详细用法见上级目录。简要：

```bash
cd Mobile-O-App

# 下载检查点
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Amshaker/Mobile-O-0.5B', local_dir='checkpoints',
                  allow_patterns=['final_merged_model_23620/*'])
"

# 导出全部组件
python export_onnx.py checkpoints --skip-download --output-dir onnx_models
```

导出结果：

| 文件 | 大小 | 说明 |
|------|------|------|
| `llm.onnx` + `llm.onnx.data` | ~2.4 GB | Qwen2-0.5B，含 25 层隐层输出 |
| `transformer.onnx` + `transformer.onnx.data` | ~2.3 GB | SANA DiT |
| `vae_decoder.onnx` | ~608 MB | DC-AE VAE 解码器 |
| `connector.onnx` | ~9 MB | MobileConditioningProjector |
| `vision_encoder.onnx` | ~468 MB | MobileCLIP 视觉编码器 |
| `llm/` | < 1 MB | Qwen2 BPE tokenizer 文件 |

---

## 推理管线说明

### 文本生成图像

```
用户提示 (generate <prompt>)
    │
    ▼
Tokenizer.encodeGenerationPrompt()      # ChatML 格式化 + BPE 编码
    │
    ▼
llm.onnx (LLMWrapper)                   # 输入: [input_ids, attention_mask]
    │                                   # 输出: logits + hidden_state_0..24
    ▼
MobileConditioningConnector             # 取最后 4 层隐层
    │   connector.onnx                  # 输出: [1, seqLen, 2304] 条件向量
    ▼
DPMSolverScheduler                      # 生成噪声时间步序列
    │
    ▼
denoising loop（默认 15 步）
    │   transformer.onnx (DiTWrapper)   # 输入: latent, timestep, conditioning
    │                                   # 输出: noise_pred
    │   CFG: noise = uncond + scale*(cond - uncond)
    ▼
vae_decoder.onnx (VAEDecoderWrapper)    # 输入: latent [1,32,16,16]
    │                                   # 输出: image [1,3,512,512]
    ▼
VaePostProcessor.floatCHWtoBitmap()     # NDK NEON: [-1,1] → ARGB_8888 Bitmap
```

### 图像理解

```
用户图像 + 文本问题
    │
    ▼
ImageProcessor                          # resize → 1024×1024 → CHW float[0,1]
    │
    ▼
vision_encoder.onnx (MobileCLIPVisionTower)
    │                                   # 输出: [1, 256, 3072] patch 特征
    ▼
llm.onnx                                # 图文交织 tokens，greedy 解码
    │                                   # 最多生成 256 tokens
    ▼
Tokenizer.decode()                      # 输出文本回答
```

---

## 生成参数

在 SettingsScreen 中可调整以下参数：

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| 推理步数 | 15 | 5–50 | 更多步数质量更高但更慢 |
| 引导系数 | 1.3 | 1.0–10.0 | CFG scale，越大越贴合提示词 |
| 启用 CFG | 是 | 开/关 | 关闭后速度提升约 2× |
| 随机种子 | 随机 | 任意整数 | 固定种子可复现结果 |

---

## 常见问题

### ORT_INVALID_PROTOBUF

```
Error code - ORT_INVALID_PROTOBUF: Load model from .../llm.onnx failed:
Protobuf parsing failed.
```

**原因：** `.onnx.data` sidecar 文件缺失，或 `.onnx` 与 `.onnx.data` 不在同一目录。

**解决：** 确认推送了完整的两个文件：
```bash
adb push onnx_models/llm.onnx      /sdcard/Android/data/com.mobileo/files/models/llm.onnx
adb push onnx_models/llm.onnx.data /sdcard/Android/data/com.mobileo/files/models/llm.onnx.data
```

### 模型加载后 App 直接崩溃（OOM）

LLM 单模型约占 2.4 GB 内存，全部模型合计约 5–6 GB。低内存设备（< 8 GB RAM）会触发 OOM Killer。建议在内存 8 GB 以上的设备上运行。

### NNAPI 报错但 App 正常运行

`OnnxSessionManager` 在启用 NNAPI 失败时会自动回退到 CPU 并打印警告日志，不影响功能。

---

## 已知限制

- 模型完全加载约需 10–30 秒（取决于设备存储速度）
- 图像生成单次推理约需 30–120 秒（取决于设备 CPU/NPU）
- NNAPI 加速需要设备支持，否则自动回退到 CPU
- 不支持图像编辑（edit 模式）——当前 Android 版本暂未实现，仅 iOS 版本支持
