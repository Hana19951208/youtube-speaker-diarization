"""
SRT and JSON output writer module.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
    if seconds is None or seconds < 0:
        seconds = 0
    
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def clean_text(text: str) -> str:
    """
    Clean text for subtitle output.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove duplicate punctuation
    import re
    text = re.sub(r'([。！？；，.!?;,])\1+', r'\1', text)
    
    return text.strip()


def write_srt(
    sentences: List[Dict[str, Any]],
    output_path: str,
    speaker_mapping: Optional[Dict[str, str]] = None,
    target_speaker: Optional[str] = None,
) -> str:
    """
    Write sentences to SRT subtitle file.
    
    Args:
        sentences: List of sentence dictionaries
        output_path: Output SRT file path
        speaker_mapping: Optional mapping of internal speaker IDs to output labels
        target_speaker: Speaker ID to label as "TARGET"
        
    Returns:
        Path to written SRT file
    """
    if speaker_mapping is None:
        speaker_mapping = {}
    
    with open(output_path, "w", encoding="utf-8") as f:
        subtitle_index = 1
        
        for sentence in sentences:
            text = sentence.get("text", "").strip()
            if not text:
                continue
            
            start = sentence.get("start", 0)
            end = sentence.get("end", 0)
            speaker = sentence.get("speaker", "unknown")
            
            # Map speaker
            if target_speaker and speaker == target_speaker:
                label = "TARGET"
            else:
                label = speaker_mapping.get(speaker, speaker)
            
            # Clean text
            text = clean_text(text)
            if not text:
                continue
            
            # Write subtitle entry
            f.write(f"{subtitle_index}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"{label}: {text}\n")
            f.write("\n")
            
            subtitle_index += 1
    
    logger.info(f"SRT file written: {output_path} ({subtitle_index - 1} subtitles)")
    return output_path


def write_json(
    data: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Write data to JSON file.
    
    Args:
        data: Data dictionary
        output_path: Output JSON file path
        
    Returns:
        Path to written JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"JSON file written: {output_path}")
    return output_path


def create_output_summary(
    sentences: List[Dict[str, Any]],
    diarization_result: Dict[str, Any],
    speaker_match_result: Optional[Dict[str, Any]],
    target_speaker: Optional[str],
    processing_times: Dict[str, float],
    audio_duration: float,
) -> Dict[str, Any]:
    """
    Create a summary of the processing results.
    
    Args:
        sentences: List of processed sentences
        diarization_result: Diarization result
        speaker_match_result: Speaker matching result
        target_speaker: Identified target speaker
        processing_times: Dictionary of processing step times
        audio_duration: Total audio duration
        
    Returns:
        Summary dictionary
    """
    # Count sentences per speaker
    speaker_sentence_counts = {}
    for sentence in sentences:
        speaker = sentence.get("speaker", "unknown")
        speaker_sentence_counts[speaker] = speaker_sentence_counts.get(speaker, 0) + 1
    
    # Calculate total processing time
    total_processing_time = sum(processing_times.values())
    
    summary = {
        "audio_info": {
            "duration_seconds": audio_duration,
            "duration_formatted": str(timedelta(seconds=int(audio_duration))),
        },
        "processing_summary": {
            "total_sentences": len(sentences),
            "total_speakers": len(diarization_result.get("speakers", [])),
            "speaker_sentence_counts": speaker_sentence_counts,
            "target_speaker": target_speaker,
        },
        "timing": {
            "total_processing_seconds": total_processing_time,
            "breakdown": processing_times,
        },
        "diarization_info": {
            "speaker_stats": diarization_result.get("speaker_stats", {}),
        },
    }
    
    if speaker_match_result:
        summary["speaker_match"] = {
            speaker: {
                "is_target": info["is_target"],
                "avg_similarity": info["avg_similarity"],
                "max_similarity": info["max_similarity"],
            }
            for speaker, info in speaker_match_result.items()
        }
    
    return summary


if __name__ == "__main__":
    print("SRT writer module loaded successfully")
