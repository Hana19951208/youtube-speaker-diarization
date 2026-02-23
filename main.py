#!/usr/bin/env python3
"""
YouTube Speaker Diarization Pipeline - Main Entry Point

This script provides a complete end-to-end pipeline for:
1. Downloading audio from YouTube videos
2. Transcribing speech using WhisperX
3. Performing speaker diarization using PyAnnote
4. Identifying target speakers using reference audio
5. Generating SRT subtitles with speaker labels

Usage:
    python main.py --youtube_url "https://www.youtube.com/watch?v=..." --ref_audio path/to/ref.wav

Environment Variables:
    HF_TOKEN: HuggingFace token (required for PyAnnote models)
"""

import sys

# Check Python version
if sys.version_info < (3, 8):
    print("Error: Python 3.8 or higher is required")
    sys.exit(1)

try:
    from pipeline import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nOperation cancelled by user")
    sys.exit(0)
except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
