"""
Speaker matching module for reference-based speaker identification.
"""
import os
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerMatcher:
    """
    Match speakers from diarization to a reference audio using embeddings.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb",
        similarity_threshold: float = 0.25,
    ):
        """
        Initialize speaker matcher.
        
        Args:
            device: Device to use (cuda, cpu, or None for auto)
            embedding_model: Embedding model to use
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.embedding_model = None
        self.reference_embedding = None
        
        logger.info(f"Speaker matcher initialized: model={embedding_model}, device={self.device}")
    
    def load_embedding_model(self):
        """Load the speaker embedding model."""
        if self.embedding_model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source=self.embedding_model_name,
                run_opts={"device": str(self.device)},
            )
            self.embedding_model.eval()
            
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback: use simple spectrogram features
            logger.warning("Falling back to simple spectral features")
            self.embedding_model = "spectral_fallback"
    
    def extract_embedding(self, audio_path: str) -> np.ndarray:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Speaker embedding vector
        """
        self.load_embedding_model()
        
        if self.embedding_model == "spectral_fallback":
            return self._extract_spectral_features(audio_path)
        
        try:
            # Load audio via librosa to avoid torchaudio binary mismatch on Kaggle
            import librosa

            y, fs = librosa.load(audio_path, sr=16000, mono=True)
            signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Extract embedding
            with torch.no_grad():
                embeddings = self.embedding_model.encode_batch(signal)
                embedding = embeddings[0].cpu().numpy()

            embedding = embedding / np.linalg.norm(embedding)
            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            logger.warning("Falling back to spectral features")
            return self._extract_spectral_features(audio_path)
    
    def _extract_spectral_features(self, audio_path: str) -> np.ndarray:
        """
        Extract simple spectral features as fallback.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Spectral feature vector
        """
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Extract features
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        # Concatenate features
        features = np.concatenate([
            mfcc_mean, mfcc_std,
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
        ])
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def set_reference(self, ref_audio_path: str):
        """
        Set reference audio for speaker matching.
        
        Args:
            ref_audio_path: Path to reference audio file
        """
        logger.info(f"Extracting reference embedding from: {ref_audio_path}")
        self.reference_embedding = self.extract_embedding(ref_audio_path)
        logger.info("Reference embedding extracted successfully")
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def match_speaker(self, audio_path: str) -> Tuple[bool, float]:
        """
        Match a speaker against the reference.
        
        Args:
            audio_path: Path to audio file to match
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        if self.reference_embedding is None:
            raise ValueError("Reference embedding not set. Call set_reference() first.")
        
        # Extract embedding
        embedding = self.extract_embedding(audio_path)
        
        # Compute similarity
        similarity = self.compute_similarity(self.reference_embedding, embedding)
        
        # Determine if it's a match
        is_match = similarity >= self.similarity_threshold
        
        return is_match, similarity
    
    def identify_speakers(
        self,
        diarization_result: Dict[str, Any],
        audio_path: str,
        output_dir: str = ".",
        min_samples: int = 3,
        max_sample_duration: float = 10.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify speakers from diarization result by comparing with reference.
        
        Args:
            diarization_result: Diarization result from PyAnnoteDiarizer
            audio_path: Original audio file path
            output_dir: Directory to save speaker samples
            min_samples: Minimum number of samples per speaker
            max_sample_duration: Maximum duration of each sample in seconds
            
        Returns:
            Dictionary mapping speaker labels to identification results
        """
        from audio_utils import cut_audio_segment
        
        if self.reference_embedding is None:
            logger.warning("Reference embedding not set, cannot identify speakers")
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        
        segments = diarization_result.get("segments", [])
        
        # Group segments by speaker
        speaker_segments = defaultdict(list)
        for segment in segments:
            speaker = segment.get("speaker", "unknown")
            speaker_segments[speaker].append(segment)
        
        # Identify each speaker
        speaker_results = {}
        
        for speaker, segs in speaker_segments.items():
            logger.info(f"Processing speaker: {speaker} ({len(segs)} segments)")
            
            # Collect samples for this speaker
            samples = []
            total_duration = 0
            target_duration = 60.0  # Target 60 seconds of audio
            
            for segment in sorted(segs, key=lambda x: x["duration"], reverse=True):
                seg_start = segment["start"]
                seg_end = segment["end"]
                seg_duration = segment["duration"]
                
                # Limit sample duration
                if seg_duration > max_sample_duration:
                    seg_end = seg_start + max_sample_duration
                    seg_duration = max_sample_duration
                
                # Cut sample
                sample_path = os.path.join(output_dir, f"{speaker}_sample_{len(samples)}.wav")
                try:
                    cut_audio_segment(audio_path, seg_start, seg_end, sample_path)
                    samples.append(sample_path)
                    total_duration += seg_duration
                    
                    if total_duration >= target_duration or len(samples) >= min_samples * 2:
                        break
                except Exception as e:
                    logger.warning(f"Failed to cut sample: {e}")
                    continue
            
            if not samples:
                logger.warning(f"No samples collected for {speaker}")
                continue
            
            # Compute similarity with reference
            similarities = []
            for sample_path in samples[:min_samples]:
                try:
                    is_match, similarity = self.match_speaker(sample_path)
                    similarities.append(similarity)
                except Exception as e:
                    logger.warning(f"Failed to match speaker: {e}")
                    continue
            
            if not similarities:
                logger.warning(f"No valid similarities for {speaker}")
                continue
            
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            
            is_target = avg_similarity >= self.similarity_threshold
            
            speaker_results[speaker] = {
                "is_target": is_target,
                "avg_similarity": float(avg_similarity),
                "max_similarity": float(max_similarity),
                "sample_count": len(similarities),
                "samples": samples,
            }
            
            logger.info(f"Speaker {speaker}: target={is_target}, avg_similarity={avg_similarity:.3f}")
        
        return speaker_results


if __name__ == "__main__":
    print("Speaker matching module loaded successfully")
