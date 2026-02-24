"""
Main pipeline for YouTube video speaker diarization and identification.
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

from audio_utils import (
    download_youtube_audio,
    transcode_to_wav,
    separate_vocals,
    check_ffmpeg,
    apply_vad,
    enhance_audio,
)
from asr_whisperx import WhisperXTranscriber, split_into_sentences
from diarization_pyannote import PyAnnoteDiarizer, assign_speakers_to_sentences
from speaker_match import SpeakerMatcher
from srt_writer import (
    write_srt,
    write_json,
    create_output_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeSpeakerPipeline:
    """
    End-to-end pipeline for YouTube video speaker diarization.
    """
    
    def __init__(
        self,
        hf_token: Optional[str] = None,
        output_dir: str = "./output",
        device: Optional[str] = None,
        whisper_model: str = "large-v3",
        max_speakers: int = 3,
        do_separation: bool = True,
        do_vad: bool = False,
        do_enhance: bool = False,
        similarity_threshold: float = 0.25,
        playlist_mode: str = "single",
    ):
        """
        Initialize the pipeline.
        
        Args:
            hf_token: HuggingFace token for gated models
            output_dir: Output directory for results
            device: Device to use (cuda, cpu, or None for auto)
            whisper_model: WhisperX model size
            max_speakers: Maximum number of speakers
            do_separation: Whether to perform vocal separation
            do_vad: Whether to apply VAD
            do_enhance: Whether to apply audio enhancement
            similarity_threshold: Threshold for speaker matching
            playlist_mode: 'single' (default) or 'all' for playlist download
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.output_dir = output_dir
        self.device = device
        self.whisper_model = whisper_model
        self.max_speakers = max_speakers
        self.do_separation = do_separation
        self.do_vad = do_vad
        self.do_enhance = do_enhance
        self.similarity_threshold = similarity_threshold
        self.playlist_mode = playlist_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.transcriber = None
        self.diarizer = None
        self.matcher = None
        
        # Timing statistics
        self.timing = {}
        
        logger.info("Pipeline initialized")
    
    def _check_prerequisites(self):
        """Check that all prerequisites are met."""
        # Check FFmpeg
        if not check_ffmpeg():
            raise RuntimeError("FFmpeg is required but not found")
        
        # Check HF token
        if not self.hf_token:
            logger.warning("HF_TOKEN not set. PyAnnote models may not be accessible.")
            logger.warning("Set HF_TOKEN environment variable or pass hf_token parameter.")
    
    def process(
        self,
        youtube_url: str,
        ref_audio_path: str,
        language: Optional[str] = None,
        target_name: str = "TARGET",
    ) -> Dict[str, Any]:
        """
        Process a YouTube video for speaker diarization.
        
        Args:
            youtube_url: YouTube video URL
            ref_audio_path: Path to reference audio for target speaker
            language: Language code (auto-detect if None)
            target_name: Label for target speaker in output
            
        Returns:
            Dictionary containing processing results
        """
        overall_start = time.time()
        
        # Check prerequisites
        self._check_prerequisites()
        
        # Step 1: Download YouTube audio
        logger.info("=" * 60)
        logger.info("Step 1: Downloading YouTube audio")
        logger.info("=" * 60)
        
        start_time = time.time()
        try:
            downloaded_audio = download_youtube_audio(
                youtube_url,
                self.output_dir,
                playlist_mode=self.playlist_mode,
            )
            logger.info(f"Downloaded: {downloaded_audio}")
        except Exception as e:
            logger.error(f"Failed to download YouTube audio: {e}")
            raise
        
        self.timing["download"] = time.time() - start_time
        
        # Step 2: Transcode to 16k mono WAV
        logger.info("=" * 60)
        logger.info("Step 2: Transcoding to 16kHz mono WAV")
        logger.info("=" * 60)
        
        start_time = time.time()
        base_name = Path(downloaded_audio).stem
        wav_path = os.path.join(self.output_dir, f"{base_name}_16k_mono.wav")
        
        try:
            transcode_to_wav(downloaded_audio, wav_path, sample_rate=16000)
        except Exception as e:
            logger.error(f"Failed to transcode audio: {e}")
            raise
        
        self.timing["transcode"] = time.time() - start_time
        
        # Step 3: Vocal separation (optional)
        processing_audio = wav_path
        
        if self.do_separation:
            logger.info("=" * 60)
            logger.info("Step 3: Vocal separation (Demucs)")
            logger.info("=" * 60)
            
            start_time = time.time()
            try:
                vocals_path = separate_vocals(wav_path, self.output_dir)
                if vocals_path != wav_path:
                    processing_audio = vocals_path
                    logger.info(f"Using vocals: {processing_audio}")
                else:
                    logger.warning("Vocal separation failed, using original audio")
            except Exception as e:
                logger.error(f"Vocal separation failed: {e}")
                logger.info("Continuing with original audio")
            
            self.timing["separation"] = time.time() - start_time
        
        # Step 4: VAD (optional)
        if self.do_vad:
            logger.info("=" * 60)
            logger.info("Step 4: Voice Activity Detection")
            logger.info("=" * 60)
            
            start_time = time.time()
            try:
                vad_path, speech_segments = apply_vad(processing_audio)
                if vad_path != processing_audio:
                    processing_audio = vad_path
                    logger.info(f"VAD applied, {len(speech_segments)} speech segments")
            except Exception as e:
                logger.error(f"VAD failed: {e}")
            
            self.timing["vad"] = time.time() - start_time
        
        # Step 5: ASR with Faster-Whisper
        logger.info("=" * 60)
        logger.info("Step 5: ASR with Faster-Whisper")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        self.transcriber = WhisperXTranscriber(
            model_size=self.whisper_model,
            device=self.device,
            language=language,
        )
        
        try:
            transcription_result = self.transcriber.transcribe(processing_audio)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
        
        self.timing["asr"] = time.time() - start_time
        
        # Prefer sentence-level segments from faster-whisper
        detected_language = transcription_result.get("language", language or "en")
        raw_segments = transcription_result.get("segments", [])

        if raw_segments:
            sentences = [
                {
                    "text": (seg.get("text") or "").strip(),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "words": seg.get("words", []),
                }
                for seg in raw_segments
                if (seg.get("text") or "").strip() and seg.get("start") is not None and seg.get("end") is not None
            ]
            words = self.transcriber.get_word_level_segments(transcription_result)
        else:
            words = self.transcriber.get_word_level_segments(transcription_result)
            sentences = split_into_sentences(words, language=detected_language)

        logger.info(f"ASR complete: {len(sentences)} sentences, {len(words)} words")
        
        # Step 6: Speaker diarization
        logger.info("=" * 60)
        logger.info("Step 6: Speaker Diarization")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        self.diarizer = PyAnnoteDiarizer(
            auth_token=self.hf_token,
            device=self.device,
            max_speakers=self.max_speakers,
        )
        
        try:
            diarization_result = self.diarizer.diarize(processing_audio)
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            # Fallback: create dummy diarization
            diarization_result = {
                "segments": [],
                "speakers": ["SPEAKER_00"],
                "speaker_stats": {"SPEAKER_00": 9999},
                "total_duration": 0,
            }
        
        self.timing["diarization"] = time.time() - start_time
        
        # Map speakers
        speaker_map = self.diarizer.map_speakers(diarization_result, max_speakers=self.max_speakers)
        
        logger.info(f"Diarization complete: {len(diarization_result['speakers'])} speakers")
        
        # Step 7: Speaker matching with reference
        logger.info("=" * 60)
        logger.info("Step 7: Reference Speaker Matching")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        self.matcher = SpeakerMatcher(
            device=self.device,
            similarity_threshold=self.similarity_threshold,
        )
        
        # Set reference audio
        try:
            self.matcher.set_reference(ref_audio_path)
        except Exception as e:
            logger.error(f"Failed to set reference audio: {e}")
            speaker_match_result = {}
            target_speaker = None
        
        # Identify speakers
        try:
            speaker_match_result = self.matcher.identify_speakers(
                diarization_result,
                processing_audio,
                output_dir=os.path.join(self.output_dir, "speaker_samples"),
            )
            
            # Find target speaker
            target_speaker = None
            for speaker, info in speaker_match_result.items():
                if info.get("is_target", False):
                    target_speaker = speaker
                    break
            
            # Update speaker map with target
            if target_speaker:
                for orig, mapped in list(speaker_map.items()):
                    if orig == target_speaker:
                        speaker_map[orig] = target_name
                        break
            
        except Exception as e:
            logger.error(f"Speaker matching failed: {e}")
            speaker_match_result = {}
            target_speaker = None
        
        self.timing["matching"] = time.time() - start_time
        
        if target_speaker:
            logger.info(f"Target speaker identified: {target_speaker}")
        else:
            logger.warning("No target speaker identified")
        
        # Step 8: Assign speakers to sentences
        logger.info("=" * 60)
        logger.info("Step 8: Assigning Speakers to Sentences")
        logger.info("=" * 60)
        
        sentences_with_speakers = assign_speakers_to_sentences(
            sentences,
            diarization_result.get("segments", []),
            speaker_map,
        )
        
        logger.info(f"Assigned speakers to {len(sentences_with_speakers)} sentences")
        
        # Step 9: Write outputs
        logger.info("=" * 60)
        logger.info("Step 9: Writing Output Files")
        logger.info("=" * 60)
        
        # Determine output filenames
        video_id = Path(youtube_url).stem if "youtube.com" in youtube_url or "youtu.be" in youtube_url else "output"
        output_prefix = os.path.join(self.output_dir, video_id)
        
        srt_path = f"{output_prefix}.srt"
        json_path = f"{output_prefix}.json"
        
        # Write SRT
        write_srt(
            sentences_with_speakers,
            srt_path,
            speaker_map,
            target_speaker,
        )
        
        # Create summary
        audio_duration = diarization_result.get("total_duration", 0)
        summary = create_output_summary(
            sentences_with_speakers,
            diarization_result,
            speaker_match_result,
            target_speaker,
            self.timing,
            audio_duration,
        )
        
        # Add detailed results to JSON
        output_data = {
            "summary": summary,
            "sentences": sentences_with_speakers,
            "diarization": {
                "segments": diarization_result.get("segments", []),
                "speakers": diarization_result.get("speakers", []),
                "speaker_stats": diarization_result.get("speaker_stats", {}),
            },
            "speaker_matching": speaker_match_result,
            "speaker_mapping": speaker_map,
            "target_speaker": target_speaker,
            "config": {
                "youtube_url": youtube_url,
                "ref_audio_path": ref_audio_path,
                "language": language,
                "max_speakers": self.max_speakers,
                "do_separation": self.do_separation,
            },
        }
        
        # Write JSON
        write_json(output_data, json_path)
        
        # Calculate total time
        overall_time = time.time() - overall_start
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output files:")
        logger.info(f"  SRT: {srt_path}")
        logger.info(f"  JSON: {json_path}")
        logger.info("")
        logger.info(f"Total duration: {audio_duration:.1f}s")
        logger.info(f"Total processing time: {overall_time:.1f}s")
        logger.info("")
        logger.info("Timing breakdown:")
        for step, duration in self.timing.items():
            logger.info(f"  {step}: {duration:.1f}s")
        logger.info("")
        logger.info("Speaker statistics:")
        for speaker, count in summary["processing_summary"]["speaker_sentence_counts"].items():
            logger.info(f"  {speaker}: {count} sentences")
        logger.info("")
        if target_speaker:
            logger.info(f"Target speaker identified: {target_speaker}")
        else:
            logger.warning("No target speaker identified")
        
        return output_data


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YouTube Speaker Diarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pipeline.py --youtube_url "https://www.youtube.com/watch?v=..." --ref_audio ref.wav
  
  # With specific options
  python pipeline.py --youtube_url "..." --ref_audio ref.wav --language zh --max_speakers 2
  
Environment Variables:
  HF_TOKEN        HuggingFace token for PyAnnote models (required)
        """
    )
    
    parser.add_argument("--youtube_url", required=True, help="YouTube video URL")
    parser.add_argument("--ref_audio", required=True, help="Path to reference audio file")
    parser.add_argument("--output_dir", default="./output", help="Output directory")
    parser.add_argument("--language", default=None, help="Language code (auto-detect if not set)")
    parser.add_argument("--max_speakers", type=int, default=3, help="Maximum number of speakers")
    parser.add_argument("--whisper_model", default="large-v3", help="WhisperX model size")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    parser.add_argument("--no_separation", action="store_true", help="Skip vocal separation")
    parser.add_argument("--vad", action="store_true", help="Apply VAD")
    parser.add_argument("--enhance", action="store_true", help="Apply audio enhancement")
    parser.add_argument("--similarity_threshold", type=float, default=0.25, help="Speaker matching threshold")
    parser.add_argument("--playlist_mode", choices=["single", "all"], default="single", help="YouTube playlist handling mode")
    
    args = parser.parse_args()
    
    # Use HF_TOKEN from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Create and run pipeline
    pipeline = YouTubeSpeakerPipeline(
        hf_token=hf_token,
        output_dir=args.output_dir,
        device=args.device,
        whisper_model=args.whisper_model,
        max_speakers=args.max_speakers,
        do_separation=not args.no_separation,
        do_vad=args.vad,
        do_enhance=args.enhance,
        similarity_threshold=args.similarity_threshold,
        playlist_mode=args.playlist_mode,
    )
    
    results = pipeline.process(
        youtube_url=args.youtube_url,
        ref_audio_path=args.ref_audio,
        language=args.language,
    )
    
    return results


if __name__ == "__main__":
    main()
