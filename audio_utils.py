"""
Audio utilities for downloading, transcoding, and processing audio files.
"""
import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from pydub import AudioSegment
import torch
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_youtube_audio(youtube_url: str, output_dir: str = ".") -> str:
    """
    Download audio from YouTube video using yt-dlp.
    
    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save the audio file
        
    Returns:
        Path to the downloaded audio file
    """
    import yt_dlp
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a template for the output filename
    output_template = os.path.join(output_dir, "%(title)s_%(id)s.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'unknown')
            video_id = info.get('id', 'unknown')
            
            # Find the downloaded file
            expected_file = os.path.join(output_dir, f"{title}_{video_id}.wav")
            
            # Handle special characters in filename
            for file in os.listdir(output_dir):
                if file.endswith('.wav') and video_id in file:
                    return os.path.join(output_dir, file)
            
            return expected_file
    except Exception as e:
        logger.error(f"Failed to download YouTube audio: {e}")
        raise


def transcode_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    Transcode audio file to 16kHz mono WAV.
    
    Args:
        input_path: Input audio file path
        output_path: Output WAV file path
        sample_rate: Target sample rate (default 16000)
        
    Returns:
        Path to the output WAV file
    """
    try:
        # Use FFmpeg for transcoding
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', str(sample_rate), '-ac', '1', '-c:a', 'pcm_s16le',
            output_path
        ]
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        
        logger.info(f"Transcoded to {output_path} ({sample_rate}Hz mono)")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg transcode failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg.")
        raise


def separate_vocals(input_path: str, output_dir: str, model: str = "htdemucs") -> str:
    """
    Separate vocals from background using Demucs.
    
    Args:
        input_path: Input audio file path
        output_dir: Output directory for separated tracks
        model: Demucs model to use (default: htdemucs)
        
    Returns:
        Path to the vocals WAV file
    """
    try:
        import demucs.separate
        
        logger.info(f"Running Demucs separation with model: {model}")
        
        # Run demucs separation
        demucs.separate.main(
            ["--two-stems", "vocals", "-n", model, "-o", output_dir, input_path]
        )
        
        # Find the vocals file
        base_name = Path(input_path).stem
        vocals_path = os.path.join(
            output_dir, model, base_name, "vocals.wav"
        )
        
        if os.path.exists(vocals_path):
            logger.info(f"Vocals separated to: {vocals_path}")
            return vocals_path
        else:
            logger.warning("Vocals separation failed, using original audio")
            return input_path
            
    except Exception as e:
        logger.error(f"Demucs separation failed: {e}")
        logger.info("Falling back to original audio without separation")
        return input_path


def load_audio(audio_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return numpy array.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        # Use torchaudio for loading
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        
        # Convert to numpy
        audio_array = waveform.squeeze().numpy()
        
        return audio_array, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        # Fallback to pydub
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(sample_rate).set_channels(1)
            audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1, 1]
            audio_array = audio_array / (2**15)
            return audio_array, sample_rate
        except Exception as e2:
            logger.error(f"Fallback audio loading also failed: {e2}")
            raise


def cut_audio_segment(audio_path: str, start: float, end: float, output_path: str):
    """
    Cut a segment from audio file.
    
    Args:
        audio_path: Input audio file
        start: Start time in seconds
        end: End time in seconds
        output_path: Output segment path
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        segment = audio[int(start*1000):int(end*1000)]
        segment.export(output_path, format="wav")
        logger.info(f"Cut segment: {start:.2f}s - {end:.2f}s -> {output_path}")
    except Exception as e:
        logger.error(f"Failed to cut segment: {e}")
        raise


def concatenate_audio_files(audio_paths: List[str], output_path: str):
    """
    Concatenate multiple audio files into one.
    
    Args:
        audio_paths: List of audio file paths
        output_path: Output concatenated file path
    """
    try:
        combined = AudioSegment.empty()
        for path in audio_paths:
            audio = AudioSegment.from_file(path)
            combined += audio
        combined.export(output_path, format="wav")
        logger.info(f"Concatenated {len(audio_paths)} files -> {output_path}")
    except Exception as e:
        logger.error(f"Failed to concatenate audio: {e}")
        raise


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and available in PATH."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("FFmpeg found")
            return True
        return False
    except FileNotFoundError:
        return False


def apply_vad(audio_path: str, output_path: str = None, min_silence_len: int = 500, silence_thresh: int = -40):
    """
    Simple silence-based VAD using pydub.

    Returns:
        (output_path, speech_segments)
    """
    from pydub.silence import detect_nonsilent

    if output_path is None:
        output_path = audio_path.replace('.wav', '_vad.wav')

    audio = AudioSegment.from_file(audio_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if not nonsilent:
        logger.warning("No speech detected by VAD; returning original audio")
        return audio_path, []

    combined = AudioSegment.empty()
    speech_segments = []
    for start_ms, end_ms in nonsilent:
        combined += audio[start_ms:end_ms]
        speech_segments.append((start_ms / 1000.0, end_ms / 1000.0))

    combined.export(output_path, format="wav")
    logger.info(f"VAD output written: {output_path}")
    return output_path, speech_segments


def enhance_audio(audio_path: str, output_path: str = None) -> str:
    """Basic audio enhancement (normalize)."""
    if output_path is None:
        output_path = audio_path.replace('.wav', '_enhanced.wav')

    audio = AudioSegment.from_file(audio_path)
    enhanced = audio.normalize()
    enhanced.export(output_path, format="wav")
    logger.info(f"Enhanced audio written: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Audio utils module loaded successfully")
