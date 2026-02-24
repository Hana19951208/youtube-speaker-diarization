"""
Speaker diarization module using PyAnnote.audio.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyAnnoteDiarizer:
    """
    Speaker diarization using PyAnnote.audio.
    """
    
    def __init__(
        self,
        auth_token: Optional[str] = None,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        max_speakers: int = 3,
    ):
        """
        Initialize PyAnnote diarizer.
        
        Args:
            auth_token: HuggingFace token for gated models
            model_name: PyAnnote model name
            device: Device to use (cuda, cpu, or None for auto)
            max_speakers: Maximum number of speakers to identify
        """
        self.model_name = model_name
        self.auth_token = auth_token or os.environ.get("HF_TOKEN")
        self.max_speakers = max_speakers
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.pipeline = None
        
        logger.info(f"PyAnnote diarizer initialized: model={model_name}, device={self.device}, max_speakers={max_speakers}")
    
    def check_auth_token(self) -> bool:
        """
        Check if HuggingFace token is available and valid.
        
        Returns:
            True if token is available, False otherwise
        """
        if not self.auth_token:
            logger.error("HuggingFace token not found!")
            logger.error("Please set HF_TOKEN environment variable or pass auth_token parameter.")
            logger.error("To get a token:")
            logger.error("  1. Visit https://huggingface.co/settings/tokens")
            logger.error("  2. Create a new token with 'read' access")
            logger.error("  3. Accept the model license at:")
            logger.error(f"     https://huggingface.co/{self.model_name}")
            return False
        
        return True
    
    def load_pipeline(self):
        """
        Load the PyAnnote diarization pipeline.
        """
        from pyannote.audio import Pipeline
        
        if not self.check_auth_token():
            raise ValueError("HuggingFace token required for PyAnnote models")
        
        logger.info(f"Loading PyAnnote pipeline: {self.model_name}")
        
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token,
            )
            self.pipeline.to(torch.device(self.device))
            
            # Set max speakers if pipeline supports it
            if hasattr(self.pipeline, 'segmentation'):
                if hasattr(self.pipeline.segmentation, 'max_speakers'):
                    self.pipeline.segmentation.max_speakers = self.max_speakers
            
            logger.info("PyAnnote pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyAnnote pipeline: {e}")
            raise
    
    def diarize(
        self,
        audio_path: str,
        min_segment_duration: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_path: Path to audio file
            min_segment_duration: Minimum segment duration in seconds
            
        Returns:
            Dictionary containing diarization results
        """
        if self.pipeline is None:
            self.load_pipeline()
        
        logger.info(f"Running diarization on: {audio_path}")
        
        try:
            # Run diarization
            diarization = self.pipeline(audio_path)
            
            # Extract segments
            segments = []
            speaker_stats = defaultdict(float)
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                duration = end - start
                
                # Skip very short segments
                if duration < min_segment_duration:
                    continue
                
                segment = {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "duration": duration,
                }
                segments.append(segment)
                speaker_stats[speaker] += duration
            
            # Sort by start time
            segments.sort(key=lambda x: x["start"])
            
            # Get unique speakers
            unique_speakers = sorted(set(s["speaker"] for s in segments))
            
            logger.info(f"Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers")
            
            return {
                "segments": segments,
                "speakers": unique_speakers,
                "speaker_stats": dict(speaker_stats),
                "total_duration": sum(speaker_stats.values()),
            }
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def map_speakers(
        self,
        diarization_result: Dict[str, Any],
        max_speakers: int = 3,
    ) -> Dict[str, str]:
        """
        Map diarization speakers to standardized labels (speaker1, speaker2, etc.).
        
        Args:
            diarization_result: Diarization result dictionary
            max_speakers: Maximum number of speakers to keep
            
        Returns:
            Dictionary mapping original speaker labels to standardized labels
        """
        speaker_stats = diarization_result.get("speaker_stats", {})
        
        # Sort speakers by total speech duration
        sorted_speakers = sorted(
            speaker_stats.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Map to standardized labels
        speaker_map = {}
        other_speakers = []
        
        for i, (speaker, duration) in enumerate(sorted_speakers):
            if i < max_speakers - 1:
                speaker_map[speaker] = f"speaker{i+1}"
            elif i == max_speakers - 1:
                speaker_map[speaker] = f"speaker{max_speakers}"
            else:
                speaker_map[speaker] = f"speaker{max_speakers}"
                other_speakers.append(speaker)
        
        # Log mapping
        for original, mapped in speaker_map.items():
            duration = speaker_stats.get(original, 0)
            logger.info(f"  {original} -> {mapped} ({duration:.1f}s)")
        
        if other_speakers:
            logger.warning(f"Speakers merged into speaker{max_speakers}: {other_speakers}")
        
        return speaker_map


def assign_speakers_to_sentences(
    sentences: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]],
    speaker_map: Dict[str, str],
    min_overlap_ratio: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Assign speaker labels to sentences based on diarization overlap.
    
    Args:
        sentences: List of sentence dictionaries
        diarization_segments: List of diarization segments
        speaker_map: Mapping from diarization speaker to output label
        min_overlap_ratio: Minimum overlap ratio to assign speaker
        
    Returns:
        List of sentences with speaker assignments
    """
    sentences_with_speakers = []
    
    for sentence in sentences:
        sent_start = sentence.get("start", 0)
        sent_end = sentence.get("end", 0)
        sent_duration = sent_end - sent_start
        
        if sent_duration <= 0:
            sentence["speaker"] = "unknown"
            sentences_with_speakers.append(sentence)
            continue
        
        # Find best matching speaker
        best_speaker = None
        best_overlap = 0
        
        for segment in diarization_segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)
            seg_speaker = segment.get("speaker", "unknown")
            
            # Calculate overlap
            overlap_start = max(sent_start, seg_start)
            overlap_end = min(sent_end, seg_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > best_overlap:
                best_overlap = overlap_duration
                best_speaker = seg_speaker
        
        # Assign speaker based on overlap ratio
        overlap_ratio = best_overlap / sent_duration if sent_duration > 0 else 0
        
        if best_speaker and overlap_ratio >= min_overlap_ratio:
            mapped_speaker = speaker_map.get(best_speaker, best_speaker)
            sentence["speaker"] = mapped_speaker
            sentence["speaker_overlap_ratio"] = overlap_ratio
        else:
            # Fallback: assign nearest diarization segment by center time (instead of unknown)
            if diarization_segments:
                sent_center = (sent_start + sent_end) / 2.0
                nearest_seg = min(
                    diarization_segments,
                    key=lambda seg: abs(((seg.get("start", 0) + seg.get("end", 0)) / 2.0) - sent_center),
                )
                nearest_speaker = nearest_seg.get("speaker", "unknown")
                sentence["speaker"] = speaker_map.get(nearest_speaker, nearest_speaker)
            else:
                sentence["speaker"] = "unknown"
            sentence["speaker_overlap_ratio"] = overlap_ratio
        
        sentences_with_speakers.append(sentence)
    
    return sentences_with_speakers


if __name__ == "__main__":
    print("ASR WhisperX module loaded successfully")
