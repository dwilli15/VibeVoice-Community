"""
Comprehensive Ebook to Audiobook Converter
Supports PDF, TXT, DOCX, EPUB formats with VibeVoice TTS
Inspired by ebook2audiobook and enhanced for VibeVoice
"""

import os
import re
import sys
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Document processing imports
try:
    import pypdf
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

# Audio processing
try:
    import soundfile as sf
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class EbookFormat(Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    EPUB = "epub"

@dataclass
class Chapter:
    """Represents a chapter in an ebook"""
    title: str
    content: str
    chapter_number: int
    word_count: int
    estimated_duration: float  # in minutes

@dataclass
class ConversionConfig:
    """Configuration for ebook conversion"""
    input_file: str
    output_dir: str
    voice_name: str = "bf_isabella"
    speed: float = 1.3
    format: str = "wav"  # wav, mp3, m4b
    chapter_break: bool = True
    max_chapter_length: int = 10000  # words
    preview_mode: bool = False
    preview_chapters: int = 2
    engine: str = "vibevoice"  # vibevoice, coqui, auto
    bitrate: str = "128k"  # For MP3/M4B encoding
    title: str = ""  # Book title for metadata
    author: str = ""  # Book author for metadata
    cover_image: str = ""  # Cover image path for M4B

class TextProcessor:
    """Advanced text processing for audiobook conversion"""
    
    def __init__(self):
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.chapter_patterns = [
            r'^Chapter\s+\d+',
            r'^\d+\.\s+',
            r'^CHAPTER\s+[IVXLC]+',
            r'^Part\s+\d+',
            r'^\*\*\*+',
            r'^---+',
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text for TTS"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common issues
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Space after sentences
        text = re.sub(r'([a-zA-Z])([.!?])([A-Z])', r'\1\2 \3', text)  # Space between sentences
        
        # Handle abbreviations
        text = re.sub(r'\bDr\.', 'Doctor', text)
        text = re.sub(r'\bMr\.', 'Mister', text)
        text = re.sub(r'\bMrs\.', 'Missus', text)
        text = re.sub(r'\bMs\.', 'Miss', text)
        
        # Handle numbers
        text = re.sub(r'\b(\d+)st\b', r'\1st', text)
        text = re.sub(r'\b(\d+)nd\b', r'\1nd', text)
        text = re.sub(r'\b(\d+)rd\b', r'\1rd', text)
        text = re.sub(r'\b(\d+)th\b', r'\1th', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better TTS processing"""
        sentences = self.sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect_chapters(self, text: str) -> List[Tuple[int, str, str]]:
        """Detect chapter boundaries in text"""
        lines = text.split('\n')
        chapters = []
        current_chapter = []
        chapter_num = 1
        chapter_title = f"Chapter {chapter_num}"
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line matches chapter pattern
            is_chapter = False
            for pattern in self.chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    is_chapter = True
                    break
            
            if is_chapter and current_chapter:
                # Save previous chapter
                content = '\n'.join(current_chapter).strip()
                if content:
                    chapters.append((chapter_num, chapter_title, content))
                
                # Start new chapter
                chapter_num += 1
                chapter_title = line if line else f"Chapter {chapter_num}"
                current_chapter = []
            else:
                current_chapter.append(line)
        
        # Add final chapter
        if current_chapter:
            content = '\n'.join(current_chapter).strip()
            if content:
                chapters.append((chapter_num, chapter_title, content))
        
        return chapters
    
    def estimate_reading_time(self, text: str, wpm: int = 180) -> float:
        """Estimate reading time in minutes"""
        word_count = len(text.split())
        return word_count / wpm

class DocumentExtractor:
    """Extract text from various document formats"""
    
    def __init__(self):
        self.text_processor = TextProcessor()
        
    def extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("pypdf not available for PDF processing")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
        except Exception as e:
            raise RuntimeError(f"Failed to extract PDF: {e}")
        
        return text
    
    def extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise RuntimeError("Could not decode text file")
    
    def extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available for DOCX processing")
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract DOCX: {e}")
    
    def extract_from_epub(self, file_path: str) -> str:
        """Extract text from EPUB file"""
        if not EPUB_AVAILABLE:
            raise ImportError("ebooklib not available for EPUB processing")
        
        try:
            book = epub.read_epub(file_path)
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    raw = item.get_content()
                    # Robust decode: try utf-8, then common Windows/Latin fallbacks, then replacement
                    try:
                        content = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        for enc in ('cp1252', 'latin-1', 'iso-8859-1'):
                            try:
                                content = raw.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            content = raw.decode('utf-8', errors='replace')
                    # Parse HTML content
                    soup = BeautifulSoup(content, 'html.parser')
                    text += soup.get_text() + "\n\n"
            
            return text
        except Exception as e:
            raise RuntimeError(f"Failed to extract EPUB: {e}")
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from any supported format"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.extract_from_pdf(str(file_path))
        elif extension == '.txt':
            return self.extract_from_txt(str(file_path))
        elif extension == '.docx':
            return self.extract_from_docx(str(file_path))
        elif extension == '.epub':
            return self.extract_from_epub(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")

class EbookToAudiobookConverter:
    """Main converter class combining all functionality"""
    
    def __init__(self):
        self.extractor = DocumentExtractor()
        self.text_processor = TextProcessor()
        self.logger = self._setup_logging()
        
        # Initialize TTS backends with preference order
        self.simple_tts_backend = None
        self.multimodel_tts_backend = None
        
        # Try to import SimpleTTSBackend first (always available)
        try:
            from simple_tts_backend import SimpleTTSBackend
            self.simple_tts_backend = SimpleTTSBackend()
            self.logger.info("✅ SimpleTTSBackend loaded")
        except ImportError as e:
            self.logger.warning(f"SimpleTTSBackend not available: {e}")
        
        # Try to import MultiModelTTSBackend for Coqui support
        try:
            from tts_backend import MultiModelTTSBackend
            self.multimodel_tts_backend = MultiModelTTSBackend()
            self.logger.info("✅ MultiModelTTSBackend loaded")
        except ImportError as e:
            self.logger.warning(f"MultiModelTTSBackend not available: {e}")
        
        # Check ffmpeg availability for packaging
        self.ffmpeg_available = self._check_ffmpeg()
        if self.ffmpeg_available:
            self.logger.info("✅ FFmpeg available for MP3/M4B packaging")
        else:
            self.logger.warning("❌ FFmpeg not available - only WAV output supported")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for conversion process"""
        logger = logging.getLogger('EbookConverter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False
    
    def _get_tts_backend(self, engine: str):
        """Get appropriate TTS backend based on engine preference"""
        if engine == "coqui" and self.multimodel_tts_backend:
            return self.multimodel_tts_backend
        elif engine == "vibevoice" and self.simple_tts_backend:
            return self.simple_tts_backend
        elif engine == "auto":
            # Auto-select best available backend
            if self.multimodel_tts_backend:
                return self.multimodel_tts_backend
            elif self.simple_tts_backend:
                return self.simple_tts_backend
        
        # Fallback to SimpleTTSBackend if available
        if self.simple_tts_backend:
            return self.simple_tts_backend
        
        return None
    
    def analyze_ebook(self, file_path: str) -> Dict:
        """Analyze ebook and return metadata"""
        self.logger.info(f"📖 Analyzing ebook: {file_path}")
        
        # Extract text
        text = self.extractor.extract_text(file_path)
        
        # Detect chapters
        chapters = self.text_processor.detect_chapters(text)
        
        # Calculate statistics
        total_words = len(text.split())
        estimated_duration = self.text_processor.estimate_reading_time(text)
        
        analysis = {
            'file_path': file_path,
            'total_words': total_words,
            'total_chapters': len(chapters),
            'estimated_duration_minutes': estimated_duration,
            'chapters': []
        }
        
        for chapter_num, title, content in chapters:
            word_count = len(content.split())
            duration = self.text_processor.estimate_reading_time(content)
            
            analysis['chapters'].append({
                'number': chapter_num,
                'title': title,
                'word_count': word_count,
                'estimated_duration_minutes': duration
            })
        
        return analysis
    
    def get_chapter_selection_options(self, file_path: str) -> Dict:
        """Get chapter selection options for interactive chapter picking"""
        self.logger.info(f"📋 Getting chapter selection for: {file_path}")
        
        # Extract text
        text = self.extractor.extract_text(file_path)
        
        # Detect chapters
        chapters = self.text_processor.detect_chapters(text)
        
        chapter_options = []
        for chapter_num, title, content in chapters:
            word_count = len(content.split())
            duration = self.text_processor.estimate_reading_time(content)
            
            # Create preview of content (first 100 words)
            content_preview = ' '.join(content.split()[:100])
            if len(content.split()) > 100:
                content_preview += "..."
            
            chapter_options.append({
                'number': chapter_num,
                'title': title,
                'word_count': word_count,
                'estimated_duration_minutes': duration,
                'preview': content_preview,
                'selected': True  # Default to all selected
            })
        
        return {
            'total_chapters': len(chapters),
            'chapters': chapter_options,
            'total_words': len(text.split()),
            'total_estimated_duration': sum(ch['estimated_duration_minutes'] for ch in chapter_options)
        }
    
    def convert_selected_chapters(
        self, 
        file_path: str, 
        selected_chapters: List[int],
        config: ConversionConfig
    ) -> Dict:
        """Convert only selected chapters to audiobook"""
        self.logger.info(f"🎯 Converting selected chapters: {selected_chapters}")
        
        # Extract text and get all chapters
        text = self.extractor.extract_text(file_path)
        all_chapters = self.text_processor.detect_chapters(text)
        
        # Filter chapters based on selection
        selected_chapter_data = []
        for chapter_num, title, content in all_chapters:
            if chapter_num in selected_chapters:
                selected_chapter_data.append((chapter_num, title, content))
        
        if not selected_chapter_data:
            raise ValueError("No chapters selected for conversion")
        
        self.logger.info(f"📖 Converting {len(selected_chapter_data)} selected chapters")
        
        # Use the normal conversion process but with filtered chapters
        # Get appropriate TTS backend
        tts_backend = self._get_tts_backend(config.engine)
        if not tts_backend:
            raise RuntimeError(f"No TTS backend available for engine: {config.engine}")
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'input_file': config.input_file,
            'output_dir': str(output_dir),
            'total_chapters': len(selected_chapter_data),
            'selected_chapters': selected_chapters,
            'audio_files': [],
            'errors': [],
            'engine_used': config.engine,
            'format': config.format
        }
        
        # Convert each selected chapter to audio
        chapter_files = []
        for chapter_num, title, content in selected_chapter_data:
            try:
                self.logger.info(f"🎙️ Converting Chapter {chapter_num}: {title}")
                
                # Clean content
                clean_content = self.text_processor.clean_text(content)
                
                if not clean_content.strip():
                    self.logger.warning(f"⚠️ Skipping empty chapter {chapter_num}")
                    continue
                
                # Generate audio filename (always WAV initially)
                safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                audio_filename = f"chapter_{chapter_num:02d}_{safe_title}.wav"
                audio_path = output_dir / audio_filename
                
                # Convert to audio using selected TTS backend
                success = self._convert_chapter_to_audio_real(
                    clean_content, 
                    str(audio_path), 
                    config,
                    tts_backend
                )
                
                if success:
                    chapter_files.append({
                        'path': str(audio_path),
                        'title': title,
                        'number': chapter_num
                    })
                    self.logger.info(f"✅ Saved: {audio_filename}")
                else:
                    results['errors'].append(f"Failed to convert chapter {chapter_num}")
                
            except Exception as e:
                error_msg = f"Error converting chapter {chapter_num}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Package audio files based on format
        if chapter_files:
            if config.format.lower() == "wav":
                # Keep individual WAV files
                results['audio_files'] = [f['path'] for f in chapter_files]
            elif config.format.lower() == "mp3":
                # Create MP3 files and combined MP3
                mp3_files = self._create_mp3_files(chapter_files, output_dir, config)
                results['audio_files'] = mp3_files
            elif config.format.lower() == "m4b":
                # Create M4B audiobook
                m4b_file = self._create_m4b_audiobook(chapter_files, output_dir, config)
                if m4b_file:
                    results['audio_files'] = [m4b_file]
                else:
                    # Fallback to MP3 if M4B fails
                    mp3_files = self._create_mp3_files(chapter_files, output_dir, config)
                    results['audio_files'] = mp3_files
                    results['errors'].append("M4B creation failed, created MP3 files instead")
        
        return results
    
    def convert_to_audiobook(self, config: ConversionConfig) -> Dict:
        """Convert ebook to audiobook"""
        self.logger.info(f"🎧 Converting {config.input_file} to audiobook")
        
        # Get appropriate TTS backend
        tts_backend = self._get_tts_backend(config.engine)
        if not tts_backend:
            raise RuntimeError(f"No TTS backend available for engine: {config.engine}")
        
        self.logger.info(f"🎙️ Using TTS engine: {config.engine}")
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and analyze text
        text = self.extractor.extract_text(config.input_file)
        chapters = self.text_processor.detect_chapters(text)
        
        if config.preview_mode:
            chapters = chapters[:config.preview_chapters]
            self.logger.info(f"🔍 Preview mode: Converting {len(chapters)} chapters")
        
        results = {
            'input_file': config.input_file,
            'output_dir': str(output_dir),
            'total_chapters': len(chapters),
            'audio_files': [],
            'errors': [],
            'engine_used': config.engine,
            'format': config.format
        }
        
        # Convert each chapter to WAV first
        chapter_files = []
        for chapter_num, title, content in chapters:
            try:
                self.logger.info(f"🎙️ Converting Chapter {chapter_num}: {title}")
                
                # Clean content
                clean_content = self.text_processor.clean_text(content)
                
                if not clean_content.strip():
                    self.logger.warning(f"⚠️ Skipping empty chapter {chapter_num}")
                    continue
                
                # Generate audio filename (always WAV initially)
                safe_title = re.sub(r'[^\w\s-]', '', title)[:50]
                audio_filename = f"chapter_{chapter_num:02d}_{safe_title}.wav"
                audio_path = output_dir / audio_filename
                
                # Convert to audio using selected TTS backend
                success = self._convert_chapter_to_audio_real(
                    clean_content, 
                    str(audio_path), 
                    config,
                    tts_backend
                )
                
                if success:
                    chapter_files.append({
                        'path': str(audio_path),
                        'title': title,
                        'number': chapter_num
                    })
                    self.logger.info(f"✅ Saved: {audio_filename}")
                else:
                    results['errors'].append(f"Failed to convert chapter {chapter_num}")
                
            except Exception as e:
                error_msg = f"Error converting chapter {chapter_num}: {e}"
                self.logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Package audio files based on format
        if chapter_files:
            if config.format.lower() == "wav":
                # Keep individual WAV files
                results['audio_files'] = [f['path'] for f in chapter_files]
            elif config.format.lower() == "mp3":
                # Create MP3 files and combined MP3
                mp3_files = self._create_mp3_files(chapter_files, output_dir, config)
                results['audio_files'] = mp3_files
            elif config.format.lower() == "m4b":
                # Create M4B audiobook with chapters
                m4b_file = self._create_m4b_audiobook(chapter_files, output_dir, config)
                if m4b_file:
                    results['audio_files'] = [m4b_file]
        
        return results
    
    def _convert_chapter_to_audio_real(self, text: str, output_path: str, 
                                     config: ConversionConfig, tts_backend) -> bool:
        """Convert single chapter to audio using real TTS"""
        try:
            # Split into manageable chunks for TTS
            sentences = self.text_processor.split_into_sentences(text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            # Build chunks of ~500 words each
            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length > 500:  # Max 500 words per chunk
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Generate audio for each chunk and collect files
            chunk_files = []
            temp_dir = Path(output_path).parent / "temp_chunks"
            temp_dir.mkdir(exist_ok=True)
            
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"🔊 Processing chunk {i+1}/{len(chunks)}")
                
                chunk_file = temp_dir / f"chunk_{i:03d}.wav"
                
                # Use TTS backend to generate speech
                success = tts_backend.generate_speech(
                    chunk, 
                    config.voice_name, 
                    str(chunk_file), 
                    speed=config.speed
                )
                
                if success and chunk_file.exists():
                    chunk_files.append(str(chunk_file))
                else:
                    self.logger.warning(f"Failed to generate chunk {i+1}")
            
            # Combine chunks using ffmpeg if available
            if chunk_files and self.ffmpeg_available:
                success = self._combine_audio_files(chunk_files, output_path)
            elif chunk_files and AUDIO_AVAILABLE:
                # Fallback: combine using soundfile
                success = self._combine_audio_files_numpy(chunk_files, output_path)
            else:
                # Create silent placeholder
                success = self._create_silent_audio(text, output_path)
            
            # Cleanup temp files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to convert chapter: {e}")
            return False
    
    def _combine_audio_files(self, audio_files: List[str], output_path: str) -> bool:
        """Combine audio files using ffmpeg"""
        try:
            import subprocess
            
            # Create concat file for ffmpeg
            concat_file = Path(output_path).parent / "concat_list.txt"
            with open(concat_file, 'w', encoding='utf-8', errors='replace') as f:
                for audio_file in audio_files:
                    f.write(f"file '{audio_file}'\n")
            
            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup concat file
            concat_file.unlink(missing_ok=True)
            
            return result.returncode == 0 and Path(output_path).exists()
            
        except Exception as e:
            self.logger.error(f"FFmpeg combination failed: {e}")
            return False
    
    def _combine_audio_files_numpy(self, audio_files: List[str], output_path: str) -> bool:
        """Combine audio files using numpy/soundfile (fallback)"""
        try:
            if not AUDIO_AVAILABLE:
                return False
            
            combined_audio = []
            sample_rate = 22050
            
            for audio_file in audio_files:
                try:
                    audio_data, sr = sf.read(audio_file)
                    if sr != sample_rate:
                        # Simple resampling (not perfect but workable)
                        audio_data = np.interp(
                            np.linspace(0, len(audio_data), int(len(audio_data) * sample_rate / sr)),
                            np.arange(len(audio_data)),
                            audio_data
                        )
                    combined_audio.append(audio_data)
                except Exception as e:
                    self.logger.warning(f"Skipping corrupt audio file {audio_file}: {e}")
            
            if combined_audio:
                final_audio = np.concatenate(combined_audio)
                sf.write(output_path, final_audio, sample_rate)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Numpy combination failed: {e}")
            return False
    
    def _create_silent_audio(self, text: str, output_path: str) -> bool:
        """Create silent audio as fallback"""
        try:
            if not AUDIO_AVAILABLE:
                # Create empty file
                Path(output_path).touch()
                return True
            
            # Create silent audio based on text length
            word_count = len(text.split())
            duration = word_count * 0.4  # Rough estimate: 0.4 seconds per word
            sample_rate = 22050
            samples = int(duration * sample_rate)
            
            # Generate silent audio
            audio_data = np.zeros(samples, dtype=np.float32)
            sf.write(output_path, audio_data, sample_rate)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Silent audio creation failed: {e}")
            return False
    
    def _create_mp3_files(self, chapter_files: List[Dict], output_dir: Path, 
                         config: ConversionConfig) -> List[str]:
        """Convert WAV files to MP3 and create combined MP3"""
        mp3_files = []
        
        if not self.ffmpeg_available:
            self.logger.warning("FFmpeg not available - cannot create MP3 files")
            return [f['path'] for f in chapter_files]  # Return WAV files
        
        try:
            import subprocess
            
            # Convert individual chapters to MP3
            for chapter in chapter_files:
                wav_path = chapter['path']
                mp3_path = str(Path(wav_path).with_suffix('.mp3'))
                
                cmd = [
                    'ffmpeg', '-y', '-i', wav_path,
                    '-codec:a', 'libmp3lame',
                    '-b:a', config.bitrate,
                    '-metadata', f"title={chapter['title']}",
                    '-metadata', f"track={chapter['number']}",
                    mp3_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    mp3_files.append(mp3_path)
                    self.logger.info(f"✅ Created MP3: {Path(mp3_path).name}")
                else:
                    self.logger.error(f"Failed to create MP3 for {chapter['title']}")
            
            # Create combined MP3 audiobook
            if mp3_files:
                combined_mp3 = output_dir / f"{Path(config.input_file).stem}_audiobook.mp3"
                concat_file = output_dir / "mp3_concat.txt"
                
                with open(concat_file, 'w', encoding='utf-8', errors='replace') as f:
                    for mp3_file in mp3_files:
                        f.write(f"file '{mp3_file}'\n")
                
                cmd = [
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', str(concat_file),
                    '-c', 'copy',
                    '-metadata', f"title={config.title or Path(config.input_file).stem}",
                    '-metadata', f"artist={config.author or 'Unknown'}",
                    '-metadata', 'genre=Audiobook',
                    str(combined_mp3)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                concat_file.unlink(missing_ok=True)
                
                if result.returncode == 0:
                    mp3_files.insert(0, str(combined_mp3))  # Add combined file first
                    self.logger.info(f"✅ Created combined MP3: {combined_mp3.name}")
            
            return mp3_files
            
        except Exception as e:
            self.logger.error(f"MP3 creation failed: {e}")
            return [f['path'] for f in chapter_files]  # Return original WAV files
    
    def _create_m4b_audiobook(self, chapter_files: List[Dict], output_dir: Path, 
                             config: ConversionConfig) -> Optional[str]:
        """Create M4B audiobook with chapter markers"""
        if not self.ffmpeg_available:
            self.logger.warning("FFmpeg not available - cannot create M4B file")
            return None
        
        try:
            import subprocess
            
            # Create M4B filename
            m4b_file = output_dir / f"{Path(config.input_file).stem}_audiobook.m4b"
            
            # Create ffmpeg metadata file with chapter information
            metadata_file = output_dir / "FFMETADATA"
            self._create_ffmetadata_file(chapter_files, metadata_file, config)
            
            # Create concat file for input
            concat_file = output_dir / "m4b_concat.txt"
            with open(concat_file, 'w', encoding='utf-8', errors='replace') as f:
                for chapter in chapter_files:
                    f.write(f"file '{chapter['path']}'\n")
            
            # Build ffmpeg command for M4B creation
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', str(concat_file),
                '-i', str(metadata_file),
                '-map_metadata', '1',
                '-codec:a', 'aac',
                '-b:a', config.bitrate,
                '-metadata', f"title={config.title or Path(config.input_file).stem}",
                '-metadata', f"artist={config.author or 'Unknown'}",
                '-metadata', 'genre=Audiobook',
                '-metadata', f"album={config.title or Path(config.input_file).stem}",
            ]
            
            # Add cover art if available
            if config.cover_image and Path(config.cover_image).exists():
                cmd.extend(['-i', config.cover_image, '-map', '2', '-c:v', 'copy', '-disposition:v', 'attached_pic'])
            
            cmd.append(str(m4b_file))
            
            # Execute ffmpeg command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temp files
            concat_file.unlink(missing_ok=True)
            metadata_file.unlink(missing_ok=True)
            
            if result.returncode == 0 and m4b_file.exists():
                self.logger.info(f"✅ Created M4B audiobook: {m4b_file.name}")
                return str(m4b_file)
            else:
                self.logger.error(f"M4B creation failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"M4B creation failed: {e}")
            return None
    
    def _create_ffmetadata_file(self, chapter_files: List[Dict], metadata_file: Path, 
                               config: ConversionConfig):
        """Create FFmpeg metadata file with chapter markers"""
        try:
            # Calculate chapter timestamps by getting duration of each WAV file
            chapter_markers = []
            current_time_ms = 0
            
            for chapter in chapter_files:
                wav_path = chapter['path']
                
                # Get duration using ffprobe if available, otherwise estimate
                try:
                    import subprocess
                    cmd = [
                        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', wav_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    duration_seconds = float(result.stdout.strip())
                    duration_ms = int(duration_seconds * 1000)
                except:
                    # Fallback: estimate duration from file size (very rough)
                    try:
                        file_size = Path(wav_path).stat().st_size
                        # Rough estimate: 44100 Hz * 2 bytes * 2 channels = 176400 bytes/sec
                        duration_seconds = file_size / 176400
                        duration_ms = int(duration_seconds * 1000)
                    except:
                        duration_ms = 60000  # 1 minute fallback
                
                chapter_markers.append({
                    'title': chapter['title'],
                    'start': current_time_ms,
                    'end': current_time_ms + duration_ms
                })
                
                current_time_ms += duration_ms
            
            # Write FFMETADATA file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(";FFMETADATA1\n")
                f.write(f"title={config.title or Path(config.input_file).stem}\n")
                f.write(f"artist={config.author or 'Unknown'}\n")
                f.write("genre=Audiobook\n")
                f.write(f"album={config.title or Path(config.input_file).stem}\n")
                f.write("\n")
                
                # Add chapters
                for i, marker in enumerate(chapter_markers):
                    f.write("[CHAPTER]\n")
                    f.write("TIMEBASE=1/1000\n")
                    f.write(f"START={marker['start']}\n")
                    f.write(f"END={marker['end']}\n")
                    f.write(f"title={marker['title']}\n")
                    f.write("\n")
                    
        except Exception as e:
            self.logger.error(f"Failed to create metadata file: {e}")
            # Create minimal metadata file
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(";FFMETADATA1\n")
                f.write(f"title={config.title or Path(config.input_file).stem}\n")
                f.write(f"artist={config.author or 'Unknown'}\n")
                f.write("genre=Audiobook\n")

def main():
    """CLI interface for ebook conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert ebooks to audiobooks using VibeVoice")
    parser.add_argument("input", help="Input ebook file (PDF, TXT, DOCX, EPUB)")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-v", "--voice", default="bf_isabella", help="Voice name")
    parser.add_argument("-s", "--speed", type=float, default=1.3, help="Speech speed")
    parser.add_argument("-f", "--format", choices=["wav", "mp3", "m4b"], default="wav", help="Audio format")
    parser.add_argument("-e", "--engine", choices=["vibevoice", "coqui", "auto"], default="vibevoice", help="TTS engine")
    parser.add_argument("-b", "--bitrate", default="128k", help="Audio bitrate for MP3/M4B")
    parser.add_argument("-t", "--title", help="Book title for metadata")
    parser.add_argument("-a", "--author", help="Book author for metadata")
    parser.add_argument("-c", "--cover", help="Cover image path for M4B")
    parser.add_argument("--preview", action="store_true", help="Preview mode (first 2 chapters)")
    parser.add_argument("--analyze", action="store_true", help="Analyze ebook only")
    
    args = parser.parse_args()
    
    converter = EbookToAudiobookConverter()
    
    if args.analyze:
        # Analyze mode
        analysis = converter.analyze_ebook(args.input)
        print(json.dumps(analysis, indent=2))
    else:
        # Convert mode
        config = ConversionConfig(
            input_file=args.input,
            output_dir=args.output,
            voice_name=args.voice,
            speed=args.speed,
            format=args.format,
            engine=args.engine,
            bitrate=args.bitrate,
            title=args.title or "",
            author=args.author or "",
            cover_image=args.cover or "",
            preview_mode=args.preview
        )
        
        results = converter.convert_to_audiobook(config)
        print(f"✅ Conversion complete!")
        print(f"📁 Output directory: {results['output_dir']}")
        print(f"🎵 Audio files created: {len(results['audio_files'])}")
        print(f"🎙️ Engine used: {results['engine_used']}")
        print(f"📊 Format: {results['format']}")
        
        if results['errors']:
            print(f"⚠️ Errors: {len(results['errors'])}")
            for error in results['errors']:
                print(f"  - {error}")

if __name__ == "__main__":
    main()
