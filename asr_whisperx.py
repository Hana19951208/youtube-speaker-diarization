"""
ASR module using faster-whisper (sentence-level timestamps).
Kept filename/class names for backward compatibility with existing imports.
"""
import logging
from typing import List, Dict, Any, Optional

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    """
    Compatibility wrapper that now uses faster-whisper internally.
    Returns sentence/segment-level timestamps (no word alignment dependency).
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language = language

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self.model = None
        logger.info(
            "Faster-Whisper initialized: model=%s, device=%s, compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )

    def _auto_select_model(self, _: str):
        if self.device != "cuda":
            return
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_memory_gb = free_memory / (1024 ** 3)
            logger.info("Available GPU memory: %.2f GB", free_memory_gb)
            if free_memory_gb < 6:
                self.model_size = "medium"
            elif free_memory_gb < 10:
                self.model_size = "large-v2"
            else:
                self.model_size = "large-v3"
            logger.info("Using model: %s", self.model_size)
        except Exception as e:
            logger.warning("Auto model selection failed: %s", e)

    def load_model(self):
        from faster_whisper import WhisperModel

        logger.info("Loading Faster-Whisper model: %s", self.model_size)
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Faster-Whisper model loaded")

    def transcribe(self, audio_path: str, batch_size: int = 8) -> Dict[str, Any]:
        if self.model is None:
            self._auto_select_model(audio_path)
            self.load_model()

        logger.info("Transcribing: %s", audio_path)
        segments, info = self.model.transcribe(
            audio_path,
            language=self.language,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=True,
        )

        result_segments: List[Dict[str, Any]] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            result_segments.append(
                {
                    "id": seg.id,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": text,
                    "words": [],
                }
            )

        lang = getattr(info, "language", None) or self.language or "unknown"
        logger.info("Detected language: %s", lang)
        logger.info("Transcribed %d segments", len(result_segments))

        return {
            "language": lang,
            "segments": result_segments,
        }

    def get_word_level_segments(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Sentence-level mode: return empty word list for compatibility.
        words: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            words.extend(segment.get("words", []))
        return words


def split_into_sentences(
    words: List[Dict[str, Any]],
    language: str = "en",
    gap_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Word-based sentence splitting fallback (kept for compatibility)."""
    if not words:
        return []

    if language in ["zh", "zh-cn", "zh-tw", "zh-hk"]:
        sentence_end_marks = "。！？；"
    else:
        sentence_end_marks = ".!?;"

    sentences = []
    current_sentence_words = []

    for i, word in enumerate(words):
        text = word.get("text", "").strip()
        if not text:
            continue
        current_sentence_words.append(word)

        is_sentence_end = any(text.endswith(mark) for mark in sentence_end_marks)

        if i < len(words) - 1:
            next_word = words[i + 1]
            current_end = word.get("end")
            next_start = next_word.get("start")
            if current_end and next_start and (next_start - current_end) > gap_threshold:
                is_sentence_end = True

        if i == len(words) - 1:
            is_sentence_end = True

        if is_sentence_end and current_sentence_words:
            sentence_text = " ".join(w.get("text", "").strip() for w in current_sentence_words)
            sentences.append(
                {
                    "text": sentence_text.strip(),
                    "start": current_sentence_words[0].get("start"),
                    "end": current_sentence_words[-1].get("end"),
                    "words": current_sentence_words,
                }
            )
            current_sentence_words = []

    return sentences
