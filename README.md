# YouTube Speaker Diarization Pipeline

A complete end-to-end pipeline for transcribing YouTube videos with speaker identification and reference-based target speaker matching.

## Features

- **YouTube Audio Download**: Download audio from any YouTube video using yt-dlp
- **Vocal Separation**: Optional vocals/background separation using Demucs
- **ASR with WhisperX**: High-quality transcription with word-level timestamps
- **Speaker Diarization**: Identify different speakers using PyAnnote.audio
- **Reference Speaker Matching**: Match speakers to a reference audio file
- **Subtitle Generation**: Export SRT files with speaker labels
- **Checkpoint & Resume**: Automatically save progress and resume from interruptions

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-speaker-diarization.git
cd youtube-speaker-diarization

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get update && sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg
```

### 2. HuggingFace Token Setup

The PyAnnote speaker diarization model requires a HuggingFace token:

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with 'read' access
3. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
4. Set the token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

### 3. Run the Pipeline

#### Command Line Usage

```bash
python main.py \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --ref_audio path/to/reference_speaker.wav \
  --language en \
  --max_speakers 3 \
  --output_dir ./output
```

#### Resume from Interruption

```bash
# Automatically resume from where it left off
python main.py \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --ref_audio path/to/reference_speaker.wav \
  --resume
```

#### Force Re-run

```bash
# Ignore existing cache and re-run everything
python main.py \
  --youtube_url "https://www.youtube.com/watch?v=..." \
  --ref_audio path/to/reference_speaker.wav \
  --force
```

## Google Colab / Kaggle

### Google Colab

Open the `YouTube_Speaker_Diarization.ipynb` notebook in Colab, or use this **one-click init cell**:

```python
# ===== Colab One-Click Init =====
import os, sys

REPO_URL = "https://github.com/Hana19951208/youtube-speaker-diarization.git"
REPO_DIR = "/content/youtube-speaker-diarization"
HF_TOKEN = ""  # 填你的 token

# 1) System deps
!apt-get update -y
!apt-get install -y ffmpeg

# 2) Fresh clone
%cd /content
!rm -rf {REPO_DIR}
!git clone {REPO_URL}
%cd {REPO_DIR}

# 3) Stable torch stack (fix torchvision::nms issue)
!pip uninstall -y torch torchvision torchaudio -q
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 4) Project deps
!pip install -q -r requirements.txt

# 5) Env
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

print("✅ Init done. Please Runtime -> Restart runtime once, then continue.")
```

重启 Runtime 后执行：

```python
%cd /content/youtube-speaker-diarization
from pipeline import YouTubeSpeakerPipeline
print("import ok")
```

### Kaggle

1. Upload the notebook to Kaggle
2. Enable GPU in Notebook Settings
3. Enable Session Persistence in Kaggle settings for longer retention
4. Follow the same steps as Colab

## Output Files

The pipeline generates the following files in the run directory:

- `output.srt` - Subtitle file with speaker labels
- `output.json` - Detailed results including:
  - Transcription with timestamps
  - Speaker diarization segments
  - Speaker matching scores
  - Processing statistics

## Checkpoint & Resume

The pipeline automatically saves progress at each major step:

1. **Download** - Saves YouTube audio and metadata
2. **Transcode** - Converts to 16kHz mono WAV
3. **Separation** - Extracts vocals (if enabled)
4. **ASR** - Transcribes with WhisperX
5. **Diarization** - Identifies speakers
6. **Matching** - Matches reference speaker
7. **SRT Generation** - Creates final output

If the pipeline is interrupted (Colab disconnects, Kaggle times out, etc.), simply re-run the same command with `--resume` to continue from where it left off.

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--youtube_url` | YouTube video URL (required) | - |
| `--ref_audio` | Path to reference audio file (required) | - |
| `--output_dir` | Output directory | `./output` |
| `--language` | Language code (en, zh, ja, etc.) | Auto-detect |
| `--max_speakers` | Maximum number of speakers | 3 |
| `--whisper_model` | WhisperX model size | `large-v3` |
| `--device` | Device (cuda/cpu) | Auto-detect |
| `--hf_token` | HuggingFace token | From env |
| `--no_separation` | Skip vocal separation | False |
| `--vad` | Apply VAD | False |
| `--resume` | Resume from checkpoint | True |
| `--force` | Force re-run all steps | False |

## Troubleshooting

### CUDA Out of Memory

Try using a smaller WhisperX model:
```bash
--whisper_model medium  # or small, base
```

### HF_TOKEN Error

Make sure you've:
1. Created a token at https://huggingface.co/settings/tokens
2. Accepted the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Set HF_TOKEN as an environment variable

### FFmpeg Error

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### YouTube Download Error

Some videos may be blocked or require authentication. Try:
- Using a different video URL
- Checking if the video is available in your region
- Updating yt-dlp: `pip install -U yt-dlp`

## License

MIT License - See LICENSE file for details

## Acknowledgments

- [WhisperX](https://github.com/m-bain/whisperX) for ASR and alignment
- [PyAnnote](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [Demucs](https://github.com/facebookresearch/demucs) for vocal separation
- [SpeechBrain](https://speechbrain.github.io/) for speaker embeddings
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube downloads
