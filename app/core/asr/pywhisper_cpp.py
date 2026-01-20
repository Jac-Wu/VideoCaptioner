import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Union
from io import BytesIO

from pydub import AudioSegment

from ...config import MODEL_PATH
from ..utils.logger import setup_logger
from .asr_data import ASRData, ASRDataSeg
from .base import BaseASR
from .status import ASRStatus

logger = setup_logger("pywhisper_asr")


# Global variable to store model in each worker process
_worker_model = None
_worker_model_path = None


def _init_worker(model_path, use_coreml, n_threads):
    """Initialize worker process with model.
    
    This is called once when each worker process starts.
    The model is loaded once and reused for all segments in this process.
    """
    global _worker_model, _worker_model_path
    import os
    from pywhispercpp.model import Model
    
    try:
        logger.info(f"Worker {os.getpid()}: Initializing model from {model_path}")
        
        if use_coreml:
            os.environ["WHISPER_COREML"] = "1"
        
        _worker_model = Model(model_path, n_threads=n_threads)
        _worker_model_path = model_path
        
        logger.info(f"Worker {os.getpid()}: Model loaded successfully")
    except Exception as e:
        logger.error(f"Worker {os.getpid()}: Failed to load model: {e}")
        raise


def _transcribe_segment_worker(args):
    """Standalone worker function for multiprocessing.
    
    This function reuses the model loaded in _init_worker.
    
    Args:
        args: Tuple of (segment_index, start_ms, end_ms, audio_segment_bytes, language, previous_text, context_stitching)
    
    Returns:
        dict with 'index', 'start_ms', 'end_ms', 'segments', 'error', 'text'
    """
    global _worker_model
    
    import tempfile
    import os
    from pydub import AudioSegment
    from io import BytesIO
    
    segment_index, start_ms, end_ms, audio_segment_bytes, language, previous_text, context_stitching = args
    
    try:
        if _worker_model is None:
            raise RuntimeError("Worker model not initialized")
        
        # Convert bytes back to AudioSegment
        audio_segment = AudioSegment.from_file(BytesIO(audio_segment_bytes))
        
        # Create temporary file for this segment
        fd, segment_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            # Export segment to file
            audio_segment.export(segment_path, format="wav")
            
            logger.info(f"Worker {os.getpid()}: Transcribing segment {segment_index}: {start_ms}ms - {end_ms}ms")
            
            # Prepare transcription parameters
            params = {
                "language": language,
                "print_progress": False,
                "print_realtime": False,
            }
            
            # Add initial prompt based on language and context stitching
            if language == "zh":
                base_prompt = "你好，我们需要使用简体中文，以下是普通话的句子。"
                if context_stitching and previous_text:
                    # Use last 20 characters from previous segment as context
                    context = previous_text[-20:] if len(previous_text) > 20 else previous_text
                    params["initial_prompt"] = base_prompt + " " + context
                    logger.info(f"Worker {os.getpid()}: Using context: '{context}'")
                else:
                    params["initial_prompt"] = base_prompt
            elif context_stitching and previous_text:
                # For non-Chinese languages, use previous text as context
                context = previous_text[-20:] if len(previous_text) > 20 else previous_text
                params["initial_prompt"] = context
                logger.info(f"Worker {os.getpid()}: Using context: '{context}'")
            
            # Perform transcription using the pre-loaded model
            segments = _worker_model.transcribe(segment_path, **params)
            
            # Adjust timestamps and collect results
            adjusted_segments = []
            full_text = ""
            for seg in segments:
                adjusted_start = (seg.t0 / 100.0) + (start_ms / 1000.0)
                adjusted_end = (seg.t1 / 100.0) + (start_ms / 1000.0)
                text = seg.text.strip()
                full_text += text + " "
                
                adjusted_segments.append({
                    'start': adjusted_start,
                    'end': adjusted_end,
                    'text': text
                })
            
            return {
                'index': segment_index,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'segments': adjusted_segments,
                'text': full_text.strip(),  # Return full text for context stitching
                'error': None
            }
            
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(segment_path):
                    os.unlink(segment_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Worker {os.getpid()}: Failed to transcribe segment {segment_index}: {e}")
        return {
            'index': segment_index,
            'start_ms': start_ms,
            'end_ms': end_ms,
            'segments': [],
            'text': '',
            'error': str(e)
        }


class PyWhisperCppASR(BaseASR):
    """PyWhisperCpp ASR implementation with CoreML support.

    Uses pywhispercpp Python library for local ASR processing with
    optional CoreML acceleration on macOS.
    """

    def __init__(
        self,
        audio_input: Union[str, bytes],
        language="en",
        whisper_model=None,
        use_cache: bool = False,
        need_word_time_stamp: bool = False,
        use_coreml: bool = True,
        n_threads: int = 4,
        # VAD parameters
        vad_filter: bool = False,
        vad_method: str = "silero_v4_fw",
        vad_threshold: float = 0.5,
        vad_max_workers: int = 2,
        vad_padding_ms: int = 400,
        vad_min_silence_ms: int = 1000,
        vad_context_stitching: bool = True,
    ):
        super().__init__(audio_input, use_cache)

        if isinstance(audio_input, str):
            assert os.path.exists(audio_input), f"Audio file not found: {audio_input}"
            assert audio_input.endswith(
                ".wav"
            ), f"Audio must be WAV format: {audio_input}"

        # Find model file in models directory
        if whisper_model:
            models_dir = Path(MODEL_PATH)
            model_files = list(models_dir.glob(f"*ggml*{whisper_model}*.bin"))
            if not model_files:
                raise ValueError(
                    f"Model file not found in {models_dir} for: {whisper_model}"
                )
            model_path = str(model_files[0])
            logger.info(f"Model found: {model_path}")
        else:
            raise ValueError("whisper_model cannot be empty")

        self.model_path = model_path
        self.need_word_time_stamp = need_word_time_stamp
        self.language = language
        self.use_coreml = use_coreml
        self.n_threads = n_threads

        # VAD parameters
        self.vad_filter = vad_filter
        self.vad_method = vad_method
        # Convert threshold from 0-100 to 0.0-1.0
        self.vad_threshold = vad_threshold / 100.0 if isinstance(vad_threshold, int) else vad_threshold
        self.vad_max_workers = vad_max_workers
        self.vad_padding_ms = vad_padding_ms
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_context_stitching = vad_context_stitching

        # Initialize pywhispercpp model (lazy loading)
        self.model = None
        
        # Thread lock for model access (pywhispercpp is not thread-safe)
        import threading
        self.model_lock = threading.Lock()

    def _load_model(self):
        """Lazy load the pywhispercpp model."""
        if self.model is not None:
            return

        try:
            from pywhispercpp.model import Model

            logger.info(f"Loading pywhispercpp model from: {self.model_path}")
            logger.info(f"CoreML enabled: {self.use_coreml}")
            logger.info(f"Threads: {self.n_threads}")

            # Initialize model with CoreML support if on macOS
            self.model = Model(
                self.model_path,
                n_threads=self.n_threads,
            )

            logger.info("Model loaded successfully")
        except ImportError as e:
            logger.error("pywhispercpp not installed. Install with: pip install pywhispercpp")
            raise RuntimeError(
                "pywhispercpp not installed. Please install it first."
            ) from e
        except Exception as e:
            logger.exception("Failed to load model")
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    def _make_segments(self, resp_data: str) -> List[ASRDataSeg]:
        asr_data = ASRData.from_srt(resp_data)
        # Filter out music markers
        filtered_segments = []
        for seg in asr_data.segments:
            text = seg.text.strip()
            # Keep text that doesn't start with 【, [, (, （
            if not (
                text.startswith("【")
                or text.startswith("[")
                or text.startswith("(")
                or text.startswith("（")
            ):
                filtered_segments.append(seg)
        return filtered_segments

    def _segment_audio_with_vad(self, audio_path: str) -> List[tuple]:
        """Segment audio using Silero VAD.
        
        Returns:
            List of (start_ms, end_ms) tuples representing speech segments
        """
        try:
            from pydub import AudioSegment
            import torch
            import numpy as np
            
            logger.info(f"Segmenting audio with VAD (method: {self.vad_method}, threshold: {self.vad_threshold})")
            logger.info(f"VAD config: padding={self.vad_padding_ms}ms, min_silence={self.vad_min_silence_ms}ms, context_stitching={self.vad_context_stitching}")
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono 16kHz WAV for VAD
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Convert to numpy array and then to torch tensor
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1, 1]
            if audio.sample_width == 2:  # 16-bit
                samples = samples / 32768.0
            elif audio.sample_width == 4:  # 32-bit
                samples = samples / 2147483648.0
            
            # Convert to torch tensor
            wav = torch.from_numpy(samples)
            sr = 16000
            
            # Load Silero VAD model
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            (get_speech_timestamps, _, _, _, _) = utils
            
            # Get speech timestamps with padding and silence merging
            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                threshold=self.vad_threshold,
                sampling_rate=sr,
                min_silence_duration_ms=self.vad_min_silence_ms,
                speech_pad_ms=self.vad_padding_ms,
                return_seconds=False
            )
            
            # Convert to milliseconds
            segments = []
            for ts in speech_timestamps:
                start_ms = int((ts['start'] / sr) * 1000)
                end_ms = int((ts['end'] / sr) * 1000)
                segments.append((start_ms, end_ms))
            
            logger.info(f"VAD found {len(segments)} speech segments")
            return segments
                    
        except Exception as e:
            logger.warning(f"VAD segmentation failed: {e}, falling back to full audio")
            # Return full audio as single segment
            audio = AudioSegment.from_file(audio_path)
            return [(0, len(audio))]

    def _transcribe_segment(self, segment_index: int, start_ms: int, end_ms: int, audio_segment, segment_path: str) -> dict:
        """Transcribe a single audio segment.
        
        Args:
            segment_index: Index of the segment
            start_ms: Start time in milliseconds
            end_ms: End time in milliseconds
            audio_segment: AudioSegment object
            segment_path: Path to save temporary segment file
            
        Returns:
            dict with 'index', 'start_ms', 'end_ms', 'segments', 'error'
        """
        try:
            # Export segment to temporary file
            audio_segment.export(segment_path, format="wav")
            
            logger.info(f"Transcribing segment {segment_index}: {start_ms}ms - {end_ms}ms")
            
            # Prepare transcription parameters
            params = {
                "language": self.language,
                "print_progress": False,
                "print_realtime": False,
            }
            
            if self.language == "zh":
                params["initial_prompt"] = "你好，我们需要使用简体中文，以下是普通话的句子。"
            
            # Perform transcription with lock (pywhispercpp is not thread-safe)
            with self.model_lock:
                segments = self.model.transcribe(segment_path, **params)
            
            # Adjust timestamps and collect results
            adjusted_segments = []
            for seg in segments:
                adjusted_start = (seg.t0 / 100.0) + (start_ms / 1000.0)
                adjusted_end = (seg.t1 / 100.0) + (start_ms / 1000.0)
                
                adjusted_segments.append({
                    'start': adjusted_start,
                    'end': adjusted_end,
                    'text': seg.text.strip()
                })
            
            return {
                'index': segment_index,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'segments': adjusted_segments,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Failed to transcribe segment {segment_index}: {e}")
            return {
                'index': segment_index,
                'start_ms': start_ms,
                'end_ms': end_ms,
                'segments': [],
                'error': str(e)
            }
        finally:
            # Clean up segment file
            try:
                if os.path.exists(segment_path):
                    os.unlink(segment_path)
            except:
                pass

    def _run(
        self, callback: Optional[Callable[[int, str], None]] = None, **kwargs: Any
    ) -> str:
        def _default_callback(_progress: int, _message: str) -> None:
            pass

        if callback is None:
            callback = _default_callback

        temp_file_path = None
        try:
            # Load model if not already loaded
            self._load_model()

            # Get audio file path
            if isinstance(self.audio_input, str):
                audio_path = self.audio_input
            else:
                # Convert bytes to WAV format
                # The bytes from chunked_asr are in MP3 format, need to convert to WAV
                import tempfile
                
                # Create temp file but don't delete automatically
                fd, temp_file_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)  # Close the file descriptor immediately
                
                try:
                    # Get the audio data
                    audio_data = self.file_binary if self.file_binary else b""
                    if not audio_data:
                        raise ValueError("No audio data available for transcription")
                    
                    # Convert audio bytes to WAV format using pydub
                    logger.info(f"Converting audio bytes ({len(audio_data)} bytes) to WAV format")
                    audio = AudioSegment.from_file(BytesIO(audio_data))
                    
                    # Export as WAV to the temporary file
                    audio.export(temp_file_path, format="wav")
                    audio_path = temp_file_path
                    logger.info(f"Created temporary WAV file: {audio_path}")
                except Exception as e:
                    if temp_file_path and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                    raise RuntimeError(f"Failed to create temporary audio file: {str(e)}") from e

            # Check if VAD is enabled
            if self.vad_filter:
                logger.info("VAD filter enabled, segmenting audio")
                callback(5, "Segmenting audio with VAD...")
                
                # Get speech segments from VAD
                vad_segments = self._segment_audio_with_vad(audio_path)
                
                if not vad_segments:
                    logger.warning("No speech segments found by VAD, transcribing full audio")
                    vad_segments = [(0, AudioSegment.from_file(audio_path).duration_seconds * 1000)]
                
                # Load full audio for segmentation
                from pydub import AudioSegment
                from io import BytesIO
                full_audio = AudioSegment.from_file(audio_path)
                
                # Transcribe segments concurrently using multiprocessing
                from concurrent.futures import ProcessPoolExecutor, as_completed
                import tempfile
                
                all_segments = []
                total_segments = len(vad_segments)
                completed_segments = 0
                
                # Use ProcessPoolExecutor for true parallel processing
                max_workers = min(self.vad_max_workers, total_segments)
                logger.info(f"Processing {total_segments} segments with {max_workers} workers (multiprocessing)")
                
                # Create executor with initializer to load model once per worker
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=_init_worker,
                    initargs=(self.model_path, self.use_coreml, self.n_threads)
                ) as executor:
                    # Process segments with context stitching
                    # For context stitching to work, we need to process segments in order
                    # We'll submit them in batches but wait for each batch to complete
                    
                    all_results = []
                    previous_text = ""  # Track previous segment text for context stitching
                    
                    # Process segments in order to maintain context
                    for idx, (start_ms, end_ms) in enumerate(vad_segments):
                        # Extract audio segment
                        audio_segment = full_audio[start_ms:end_ms]
                        
                        # Convert audio segment to bytes for pickling
                        buffer = BytesIO()
                        audio_segment.export(buffer, format="wav")
                        audio_segment_bytes = buffer.getvalue()
                        
                        # Prepare arguments tuple for worker (now with context stitching params)
                        args = (
                            idx,
                            start_ms,
                            end_ms,
                            audio_segment_bytes,
                            self.language,
                            previous_text,  # Pass previous segment text for context
                            self.vad_context_stitching,  # Pass context stitching flag
                        )
                        
                        # Submit task and wait for completion to get context for next segment
                        future = executor.submit(_transcribe_segment_worker, args)
                        result = future.result()  # Wait for this segment to complete
                        
                        all_results.append(result)
                        
                        # Update previous text for next segment's context
                        if self.vad_context_stitching and result['text']:
                            previous_text = result['text']
                        
                        completed_segments += 1
                        segment_progress = int(10 + (completed_segments / total_segments) * 70)
                        callback(segment_progress, f"Completed {completed_segments}/{total_segments} segments...")
                        
                        if result['error']:
                            logger.warning(f"Segment {result['index']} failed: {result['error']}")
                
                # Merge segments from all results
                for result in all_results:
                    all_segments.extend(result['segments'])
                
                callback(80, "Merging segments...")
                
                # Convert to SRT format
                srt_content = self._segments_to_srt_from_dict(all_segments)
                
            else:
                # No VAD, transcribe full audio
                logger.info(f"Transcribing audio: {audio_path}")
                callback(10, "Starting transcription...")

                # Prepare transcription parameters
                params = {
                    "language": self.language,
                    "print_progress": False,
                    "print_realtime": False,
                }

                # Add Chinese prompt if language is Chinese
                if self.language == "zh":
                    params["initial_prompt"] = "你好，我们需要使用简体中文，以下是普通话的句子。"

                callback(20, "Transcribing...")

                # Perform transcription with lock (pywhispercpp is not thread-safe)
                with self.model_lock:
                    segments = self.model.transcribe(audio_path, **params)

                callback(80, "Processing results...")

                # Convert segments to SRT format
                srt_content = self._segments_to_srt(segments)

            callback(*ASRStatus.COMPLETED.callback_tuple())
            logger.info("Transcription completed successfully")

            return srt_content

        except Exception as e:
            logger.exception("Transcription failed")
            raise RuntimeError(f"Transcription failed: {str(e)}") from e
        finally:
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

    def _segments_to_srt_from_dict(self, segments: List[dict]) -> str:
        """Convert segment dictionaries to SRT format."""
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            text = seg['text']
            
            # Skip empty segments
            if not text:
                continue
            
            # Format timestamps
            start_str = self._format_timestamp(seg['start'])
            end_str = self._format_timestamp(seg['end'])
            
            # Add SRT entry
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")
        
        return "\n".join(srt_lines)

    def _segments_to_srt(self, segments) -> str:
        """Convert pywhispercpp segments to SRT format."""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            # Get segment data
            start_time = segment.t0 / 100.0  # Convert to seconds
            end_time = segment.t1 / 100.0
            text = segment.text.strip()

            # Skip empty segments
            if not text:
                continue

            # Format timestamps
            start_str = self._format_timestamp(start_time)
            end_str = self._format_timestamp(end_time)

            # Add SRT entry
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _get_key(self):
        return f"{self.crc32_hex}-{self.need_word_time_stamp}-{self.model_path}-{self.language}-{self.use_coreml}"


if __name__ == "__main__":
    # Example usage
    asr = PyWhisperCppASR(
        audio_input="audio.wav",
        whisper_model="tiny",
        language="en",
        use_coreml=True,
        need_word_time_stamp=True,
    )
    asr_data = asr.run(callback=print)
    print(asr_data)
