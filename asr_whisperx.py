"""
ASR module using WhisperX for transcription and word-level alignment.
"""
import os
import logging
from typing import List, Dict,Tuple, Any, Optional
import torch
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    """
    WhisperX-based transcriber with word-level alignment.
    """
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Initialize WhisperX transcriber.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Device to use (cuda, cpu, or None for auto)
            compute_type: Compute type (float16, int8, float32)
            language: Language code (en, zh, etc.) or None for auto-detect
        """
        self.model_size = model_size
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Auto-detect compute type
        if compute_type is None:
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type
        
        # Model placeholders
        self.model = None
        self.align_model = None
        self.align_metadata = None
        
        logger.info(f"WhisperX initialized: model={model_size}, device={self.device}, compute_type={self.compute_type}")
    
    def _auto_select_model(self, audio_path: str):
        """
        Auto-select model based on available memory.
        """
        if self.device == "cuda":
            try:
                # Get available GPU memory
                free_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory -= torch.cuda.memory_allocated(0)
                free_memory_gb = free_memory / (1024**3)
                
                logger.info(f"Available GPU memory: {free_memory_gb:.2f} GB")
                
                # Select model based on available memory
                if free_memory_gb < 6:
                    logger.warning("Low GPU memory, downgrading to medium model")
                    self.model_size = "medium"
                elif free_memory_gb < 10:
                    logger.info("Using large-v2 model")
                    self.model_size = "large-v2"
                else:
                    logger.info("Using large-v3 model")
                    self.model_size = "large-v3"
                    
            except Exception as e:
                logger.warning(f"Failed to auto-select model: {e}")
    
    def load_model(self):
        """Load WhisperX model."""
        import whisperx
        
        logger.info(f"Loading WhisperX model: {self.model_size}")
        
        self.model = whisperx.load_model(
            self.model_size,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        
        logger.info("WhisperX model loaded successfully")
    
    def load_alignment_model(self, language: str):
        """Load alignment model for word-level timestamps."""
        import whisperx
        
        logger.info(f"Loading alignment model for language: {language}")
        
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=self.device,
        )
        
        logger.info("Alignment model loaded successfully")
    
    def transcribe(
        self,
        audio_path: str,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using WhisperX.
        
        Args:
            audio_path: Path to audio file
            batch_size: Batch size for transcription
            
        Returns:
            Dictionary containing segments with text and timestamps
        """
        import whisperx
        
        # Load model if not loaded
        if self.model is None:
            self._auto_select_model(audio_path)
            self.load_model()
        
        # Load audio
        logger.info(f"Loading audio: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        
        # Transcribe
        logger.info("Running transcription...")
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=self.language,
        )
        
        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")
        logger.info(f"Transcribed {len(result['segments'])} segments")
        
        # Align for word-level timestamps
        if self.align_model is None:
            self.load_alignment_model(detected_language)
        
        logger.info("Running alignment for word-level timestamps...")
        result = whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        
        logger.info(f"Alignment complete. Total words: {sum(len(s.get('words', [])) for s in result['segments'])}")
        
        return result
    
    def get_word_level_segments(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract word-level segments from alignment result.
        
        Args:
            result: WhisperX result with alignment
            
        Returns:
            List of word-level segments
        """
        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append({
                    "text": word.get("word", "").strip(),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "score": word.get("score"),
                })
        return words


def split_into_sentences(
    words: List[Dict[str, Any]],
    language: str = "en",
    gap_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Split word-level segments into sentences.
    
    Args:
        words: List of word-level segments
        language: Language code
        gap_threshold: Minimum gap in seconds to split sentence
        
    Returns:
        List of sentence-level segments
    """
    if not words:
        return []
    
    # Define punctuation markers for different languages
    if language in ["zh", "zh-cn", "zh-tw", "zh-hk"]:
        sentence_end_marks = "。！？；"
        pause_marks = "，、"
    else:
        sentence_end_marks = ".!?;"
        pause_marks = ",:"
    
    sentences = []
    current_sentence_words = []
    
    for i, word in enumerate(words):
        text = word.get("text", "").strip()
        if not text:
            continue
        
        current_sentence_words.append(word)
        
        # Check if this word ends a sentence
        is_sentence_end = False
        
        # Check for sentence-ending punctuation
        if any(text.endswith(mark) for mark in sentence_end_marks):
            is_sentence_end = True
        
        # Check for long gaps (if not the last word)
        if i < len(words) - 1:
            next_word = words[i + 1]
            current_end = word.get("end")
            next_start = next_word.get("start")
            
            if current_end and next_start and (next_start - current_end) > gap_threshold:
                is_sentence_end = True
        
        # Final word always ends a sentence
        if i == len(words) - 1:
            is_sentence_end = True
        
        if is_sentence_end and current_sentence_words:
            # Create sentence
            sentence_text = " ".join(w.get("text", "").strip() for w in current_sentence_words)
            start_time = current_sentence_words[0].get("start")
            end_time = current_sentence_words[-1].get("end")
            
            sentences.append({
                "text": sentence_text.strip(),
                "start": start_time,
                "end": end_time,
                "words": current_sentence_words,
            })
            
            current_sentence_words = []
    
    return sentences


def apply_vad(audio_path: str, output_path: str = None, 
              aggressiveness: int = 1) -> Tuple[str, List[Tuple[float, float]]]:
    """
    Apply Voice Activity Detection to remove silence.
    
    Args:
        audio_path: Input audio file
        output_path: Output file path (optional)
        aggressiveness: VAD aggressiveness (0-3)
        
    Returns:
        Tuple of (output_path, list of speech segments)
    """
    try:
        import webrtcvad
    except ImportError:
        logger.warning("webrtcvad not available, skipping VAD")
        return audio_path, [(0, None)]
    
    if output_path is None:
        output_path = audio_path.replace('.wav', '_vad.wav')
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)
    
    # Convert to proper format for webrtcvad
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    
    # Run VAD
    vad = webrtcvad.Vad(aggressiveness)
    
    # Process in 30ms frames
    frame_duration = 30  # ms
    bytes_per_frame = int(16000 * 2 * frame_duration / 1000)
    
    raw_data = audio.raw_data
    frames = []
    
    for i in range(0, len(raw_data), bytes_per_frame):
        frame = raw_data[i:i+bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        is_speech = vad.is_speech(frame, 16000)
        frames.append((i / 16000 / 2, is_speech))  # timestamp in seconds
    
    # Extract speech segments
    speech_segments = []
    in_speech = False
    start_time = 0
    
    for timestamp, is_speech in frames:
        if is_speech and not in_speech:
            start_time = timestamp
            in_speech = True
        elif not is_speech and in_speech:
            speech_segments.append((start_time, timestamp))
            in_speech = False
    
    if in_speech:
        speech_segments.append((start_time, None))
    
    # Export speech segments
    if speech_segments:
        output_audio = AudioSegment.empty()
        for start, end in speech_segments:
            if end is None:
                segment = audio[int(start*1000):]
            else:
                segment = audio[int(start*1000):int(end*1000)]
            output_audio += segment
        
        output_audio.export(output_path, format="wav")
        logger.info(f"VAD applied: {len(speech_segments)} speech segments, saved to {output_path}")
        return output_path, speech_segments
    
    logger.warning("No speech detected in audio")
    return audio_path, []


def enhance_audio(audio_path: str, output_path: str = None) -> str:
    """
    Apply basic audio enhancement (normalization, noise reduction if available).
    
    Args:
        audio_path: Input audio file
        output_path: Output file path
        
    Returns:
        Path to enhanced audio file
    """
    if output_path is None:
        output_path = audio_path.replace('.wav', '_enhanced.wav')
    
    try:
        # Load audio
        audio = AudioSegment.from_wav(audio_path)
        
        # Normalize
        audio = audio.normalize()
        
        # Try to apply noise reduction if available
        try:
            import noisereduce as nr
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            
            # Normalize to [-1, 1]
            samples = samples / (2**15)
            
            # Apply noise reduction
            reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)
            
            # Convert back to audio segment
            reduced_noise = (reduced_noise * (2**15)).astype(np.int16)
            audio = AudioSegment(
                reduced_noise.tobytes(),
                frame_rate=audio.frame_rate,
                sample_width=2,
                channels=1
            )
            
        except ImportError:
            logger.info("noisereduce not available, skipping noise reduction")
        
        # Export
        audio.export(output_path, format="wav")
        logger.info(f"Audio enhanced and saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio enhancement failed: {e}")
        logger.info("Returning original audio without enhancement")
        return audio_path


def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logger.info(f"FFmpeg found: {version}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error("FFmpeg not found! Please install FFmpeg:")
    logger.error("  Ubuntu/Debian: sudo apt-get install ffmpeg")
    logger.error("  macOS: brew install ffmpeg")
    logger.error("  Windows: download from https://ffmpeg.org/download.html")
    return False


if __name__ == "__main__":
    # Test FFmpeg
    check_ffmpeg()
    print("Audio utils module loaded successfully")
