"""
LM Studio Integration for VibeVoice Community
Generates multi-speaker scripts and dialogues for TTS conversion
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

@dataclass
class SpeakerProfile:
    """Defines a speaker's personality and characteristics"""
    name: str
    personality: str  # professional, casual, enthusiastic, scholarly, etc.
    background: str   # expert, student, host, guest, narrator
    speaking_style: str  # formal, conversational, animated, calm
    voice_preference: str = ""  # Suggested TTS voice
    gender: str = "neutral"
    age_range: str = "adult"
    accent: str = "neutral"

@dataclass 
class ConversationConfig:
    """Configuration for conversation generation"""
    topic: str
    speakers: List[SpeakerProfile]
    style: str = "podcast"  # podcast, interview, dialogue, lecture, debate
    duration_minutes: int = 10
    target_audience: str = "general"  # general, academic, technical, children
    tone: str = "informative"  # informative, casual, educational, entertaining
    include_intro: bool = True
    include_outro: bool = True
    chapter_breaks: bool = False
    
class LLMProvider(Enum):
    """Supported LLM providers"""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL_API = "local_api"

@dataclass
class LLMConfig:
    """Configuration for LLM connection"""
    provider: LLMProvider
    api_url: str
    api_key: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 30

class LMStudioConnector:
    """Connect to LM Studio for script generation"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.session = requests.Session()
        
        # Test connection
        if not self._test_connection():
            self.logger.warning("⚠️ LLM connection test failed - check configuration")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('LMStudio')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_default_config(self) -> LLMConfig:
        """Get default LM Studio configuration"""
        # Check environment variables for configuration
        api_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LM_STUDIO_API_KEY", "")
        model_name = os.getenv("LM_STUDIO_MODEL", "")
        
        return LLMConfig(
            provider=LLMProvider.LM_STUDIO,
            api_url=api_url,
            api_key=api_key,
            model_name=model_name
        )
    
    def _test_connection(self) -> bool:
        """Test connection to LLM provider"""
        try:
            if self.config.provider == LLMProvider.LM_STUDIO:
                # Test LM Studio endpoint
                response = self.session.get(f"{self.config.api_url}/models", timeout=5)
                return response.status_code == 200
            elif self.config.provider == LLMProvider.OLLAMA:
                # Test Ollama endpoint  
                response = self.session.get(f"{self.config.api_url}/api/tags", timeout=5)
                return response.status_code == 200
            else:
                # For API-based providers, we'll test during first generation
                return True
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def generate_conversation(self, config: ConversationConfig) -> Optional[str]:
        """Generate a multi-speaker conversation"""
        try:
            self.logger.info(f"🎭 Generating {config.style} about '{config.topic}' with {len(config.speakers)} speakers")
            
            # Build conversation prompt
            prompt = self._build_conversation_prompt(config)
            
            # Generate using LLM
            response_text = self._call_llm(prompt)
            
            if response_text:
                # Post-process the generated conversation
                formatted_conversation = self._format_conversation(response_text, config)
                self.logger.info(f"✅ Generated conversation: {len(formatted_conversation.split())} words")
                return formatted_conversation
            else:
                self.logger.error("❌ Failed to generate conversation")
                return None
                
        except Exception as e:
            self.logger.error(f"Conversation generation failed: {e}")
            return None
    
    def _build_conversation_prompt(self, config: ConversationConfig) -> str:
        """Build prompt for conversation generation"""
        # Speaker descriptions
        speaker_descriptions = []
        for speaker in config.speakers:
            desc = f"{speaker.name}: {speaker.personality} {speaker.background}, speaks in a {speaker.speaking_style} manner"
            speaker_descriptions.append(desc)
        
        speaker_list = "\n".join(speaker_descriptions)
        
        # Conversation length estimate (rough: 150 words per minute)
        target_words = config.duration_minutes * 150
        
        prompt = f"""Create a {config.style} conversation about "{config.topic}" with the following speakers:

{speaker_list}

Requirements:
- Target length: approximately {target_words} words ({config.duration_minutes} minutes)
- Tone: {config.tone}
- Audience: {config.target_audience}
- Style: {config.style}
{'- Include a natural introduction' if config.include_intro else ''}
{'- Include a natural conclusion' if config.include_outro else ''}

Format the output exactly like this example:
Speaker 1: [their dialogue here]
Speaker 2: [their response here]
Speaker 1: [continuation...]

Make the conversation natural, engaging, and informative. Each speaker should maintain their unique personality and speaking style throughout. Ensure natural turn-taking and realistic dialogue flow.

Begin the conversation:"""

        return prompt
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the configured LLM with the prompt"""
        try:
            if self.config.provider == LLMProvider.LM_STUDIO:
                return self._call_lm_studio(prompt)
            elif self.config.provider == LLMProvider.OLLAMA:
                return self._call_ollama(prompt)
            elif self.config.provider == LLMProvider.OPENAI:
                return self._call_openai(prompt)
            else:
                self.logger.error(f"Unsupported provider: {self.config.provider}")
                return None
                
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return None
    
    def _call_lm_studio(self, prompt: str) -> Optional[str]:
        """Call LM Studio API"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        if self.config.api_key:
            headers['Authorization'] = f'Bearer {self.config.api_key}'
        
        payload = {
            'model': self.config.model_name or 'local-model',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'stream': False
        }
        
        response = self.session.post(
            f"{self.config.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            self.logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
            return None
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API"""
        payload = {
            'model': self.config.model_name or 'llama2',
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': self.config.temperature,
                'num_predict': self.config.max_tokens
            }
        }
        
        response = self.session.post(
            f"{self.config.api_url}/api/generate",
            json=payload,
            timeout=self.config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
    
    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }
        
        payload = {
            'model': self.config.model_name or 'gpt-3.5-turbo',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens
        }
        
        response = self.session.post(
            'https://api.openai.com/v1/chat/completions',
            json=payload,
            headers=headers,
            timeout=self.config.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            self.logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            return None
    
    def _format_conversation(self, raw_text: str, config: ConversationConfig) -> str:
        """Format and clean the generated conversation"""
        lines = raw_text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with speaker format
            if re.match(r'^(Speaker \d+|[A-Za-z]+ \d*|[A-Za-z]+):\s*', line):
                # This is a speaker line
                formatted_lines.append(line)
            elif formatted_lines:
                # This might be a continuation of the previous speaker
                # Append to the last speaker's dialogue
                formatted_lines[-1] += ' ' + line
        
        # Clean up speaker names to match the provided speakers
        if len(config.speakers) > 0:
            formatted_lines = self._normalize_speaker_names(formatted_lines, config.speakers)
        
        return '\n'.join(formatted_lines)
    
    def _normalize_speaker_names(self, lines: List[str], speakers: List[SpeakerProfile]) -> List[str]:
        """Normalize speaker names to match the provided speaker profiles"""
        normalized_lines = []
        speaker_mapping = {}
        
        # Create a mapping from generic names to actual speaker names
        for i, speaker in enumerate(speakers):
            speaker_mapping[f"Speaker {i+1}"] = speaker.name
            speaker_mapping[f"Speaker{i+1}"] = speaker.name
        
        for line in lines:
            # Replace generic speaker names with actual names
            for generic_name, actual_name in speaker_mapping.items():
                if line.startswith(f"{generic_name}:"):
                    line = line.replace(f"{generic_name}:", f"{actual_name}:", 1)
                    break
            
            normalized_lines.append(line)
        
        return normalized_lines
    
    def convert_text_to_dialogue(self, source_text: str, speaker_count: int = 2, 
                               style: str = "educational") -> Optional[str]:
        """Convert existing text into multi-speaker dialogue"""
        try:
            self.logger.info(f"📝 Converting text to {speaker_count}-speaker dialogue")
            
            # Create default speakers for the conversion
            speakers = []
            if speaker_count == 2:
                speakers = [
                    SpeakerProfile("Alex", "knowledgeable", "expert", "clear"),
                    SpeakerProfile("Sam", "curious", "student", "conversational")
                ]
            elif speaker_count == 3:
                speakers = [
                    SpeakerProfile("Alex", "knowledgeable", "host", "professional"),
                    SpeakerProfile("Jordan", "analytical", "expert", "thoughtful"),
                    SpeakerProfile("Casey", "curious", "interviewer", "engaging")
                ]
            else:
                # Generate speakers dynamically
                for i in range(speaker_count):
                    speakers.append(SpeakerProfile(
                        f"Speaker{i+1}",
                        "informative",
                        "expert",
                        "professional"
                    ))
            
            # Build conversion prompt
            prompt = f"""Convert the following text into a natural {speaker_count}-speaker {style} dialogue. 

The speakers should discuss the content naturally, with:
- {speakers[0].name}: {speakers[0].personality} {speakers[0].background}
- {speakers[1].name}: {speakers[1].personality} {speakers[1].background}
{f"- {speakers[2].name}: {speakers[2].personality} {speakers[2].background}" if len(speakers) > 2 else ""}

Make it conversational but preserve all important information. Use natural dialogue flow with questions, explanations, and discussions.

Source text:
{source_text[:3000]}  # Limit text length for API

Format as:
Speaker1: [dialogue]
Speaker2: [response]
etc.

Convert to dialogue:"""

            response_text = self._call_llm(prompt)
            
            if response_text:
                # Create a temporary config for formatting
                temp_config = ConversationConfig(
                    topic="Text Conversion",
                    speakers=speakers,
                    style=style
                )
                formatted_dialogue = self._format_conversation(response_text, temp_config)
                self.logger.info(f"✅ Converted to dialogue: {len(formatted_dialogue.split())} words")
                return formatted_dialogue
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Text-to-dialogue conversion failed: {e}")
            return None
    
    def assign_speaker_personalities(self, script: str, voice_profiles: List[str]) -> str:
        """Assign specific voice personalities to speakers in an existing script"""
        try:
            self.logger.info(f"🎭 Assigning personalities to {len(voice_profiles)} speakers")
            
            # Parse existing script to identify speakers
            speakers_found = set()
            lines = script.split('\n')
            
            for line in lines:
                match = re.match(r'^([^:]+):', line.strip())
                if match:
                    speakers_found.add(match.group(1).strip())
            
            speakers_list = list(speakers_found)
            
            # Create personality assignment prompt
            personality_assignments = []
            for i, speaker in enumerate(speakers_list):
                if i < len(voice_profiles):
                    personality_assignments.append(f"{speaker} → {voice_profiles[i]}")
                else:
                    personality_assignments.append(f"{speaker} → neutral professional")
            
            prompt = f"""Modify the speaking style and word choice in this script to match the assigned personalities:

Personality assignments:
{chr(10).join(personality_assignments)}

Original script:
{script}

Rewrite each speaker's lines to match their assigned personality while keeping the same information and dialogue flow. Maintain the Speaker: format.

Modified script:"""

            response_text = self._call_llm(prompt)
            
            if response_text:
                self.logger.info("✅ Assigned speaker personalities")
                return response_text.strip()
            else:
                return script  # Return original if assignment fails
                
        except Exception as e:
            self.logger.error(f"Personality assignment failed: {e}")
            return script

class ScriptToAudioConverter:
    """Convert LLM-generated scripts to audiobooks"""
    
    def __init__(self, llm_connector: LMStudioConnector = None):
        self.llm_connector = llm_connector or LMStudioConnector()
        self.logger = logging.getLogger('ScriptToAudio')
    
    def generate_and_convert_podcast(self, topic: str, output_dir: str, 
                                   speakers: List[SpeakerProfile] = None,
                                   duration_minutes: int = 10,
                                   voice_engine: str = "auto") -> Dict:
        """Generate a podcast script and convert to audio"""
        try:
            # Use default speakers if none provided
            if not speakers:
                speakers = [
                    SpeakerProfile("Alex", "enthusiastic", "host", "engaging", "bf_isabella"),
                    SpeakerProfile("Jordan", "knowledgeable", "expert", "thoughtful", "af_heart")
                ]
            
            # Generate conversation
            conversation_config = ConversationConfig(
                topic=topic,
                speakers=speakers,
                style="podcast",
                duration_minutes=duration_minutes,
                include_intro=True,
                include_outro=True
            )
            
            script = self.llm_connector.generate_conversation(conversation_config)
            
            if not script:
                return {"error": "Failed to generate script"}
            
            # Save script to temporary file
            script_file = Path(output_dir) / f"generated_podcast_{topic.replace(' ', '_')}.txt"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(f"Generated Podcast: {topic}\n")
                f.write(f"Speakers: {', '.join([s.name for s in speakers])}\n\n")
                f.write(script)
            
            # Convert to audio using existing ebook converter
            from ebook_converter import EbookToAudiobookConverter, ConversionConfig
            
            config = ConversionConfig(
                input_file=str(script_file),
                output_dir=output_dir,
                voice_name=speakers[0].voice_preference or "bf_isabella",
                speed=1.2,  # Slightly faster for podcast feel
                format="mp3",
                engine=voice_engine,
                title=f"Generated Podcast: {topic}",
                author="AI Generated Content"
            )
            
            converter = EbookToAudiobookConverter()
            results = converter.convert_to_audiobook(config)
            
            # Keep the script file for reference
            results['script_file'] = str(script_file)
            results['speakers'] = [asdict(speaker) for speaker in speakers]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Podcast generation failed: {e}")
            return {"error": str(e)}
    
    def convert_existing_script(self, script_file: str, output_dir: str,
                              speaker_voice_mapping: Dict[str, str] = None,
                              voice_engine: str = "auto") -> Dict:
        """Convert an existing multi-speaker script to audio"""
        try:
            # Read script
            with open(script_file, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # Analyze speakers in the script
            speakers_found = set()
            lines = script_content.split('\n')
            
            for line in lines:
                match = re.match(r'^([^:]+):', line.strip())
                if match:
                    speakers_found.add(match.group(1).strip())
            
            self.logger.info(f"📋 Found speakers: {', '.join(speakers_found)}")
            
            # TODO: Implement speaker-specific voice assignment
            # For now, use the existing converter which will handle multi-speaker format
            
            from ebook_converter import EbookToAudiobookConverter, ConversionConfig
            
            # Use first voice from mapping or default
            primary_voice = "bf_isabella"
            if speaker_voice_mapping:
                primary_voice = list(speaker_voice_mapping.values())[0]
            
            config = ConversionConfig(
                input_file=script_file,
                output_dir=output_dir,
                voice_name=primary_voice,
                speed=1.2,
                format="mp3",
                engine=voice_engine,
                title=Path(script_file).stem,
                author="Multi-Speaker Script"
            )
            
            converter = EbookToAudiobookConverter()
            results = converter.convert_to_audiobook(config)
            
            results['speakers_found'] = list(speakers_found)
            results['speaker_voice_mapping'] = speaker_voice_mapping or {}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Script conversion failed: {e}")
            return {"error": str(e)}

def main():
    """CLI interface for LM Studio integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and convert AI scripts to audiobooks")
    parser.add_argument("command", choices=["generate", "convert"], help="Command to execute")
    parser.add_argument("-t", "--topic", help="Topic for conversation generation")
    parser.add_argument("-f", "--file", help="Script file to convert")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-d", "--duration", type=int, default=10, help="Duration in minutes")
    parser.add_argument("-s", "--speakers", type=int, default=2, help="Number of speakers")
    parser.add_argument("-v", "--voice-engine", default="auto", help="TTS engine to use")
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1", help="LM Studio URL")
    parser.add_argument("--model", help="Model name to use")
    
    args = parser.parse_args()
    
    # Setup LM Studio connection
    llm_config = LLMConfig(
        provider=LLMProvider.LM_STUDIO,
        api_url=args.lm_studio_url,
        model_name=args.model or ""
    )
    
    llm_connector = LMStudioConnector(llm_config)
    converter = ScriptToAudioConverter(llm_connector)
    
    if args.command == "generate":
        if not args.topic:
            print("❌ Topic required for generation mode")
            return
        
        # Generate default speakers
        speakers = []
        for i in range(args.speakers):
            speakers.append(SpeakerProfile(
                f"Speaker{i+1}",
                "professional",
                "expert",
                "clear"
            ))
        
        print(f"🎭 Generating {args.duration}-minute conversation about '{args.topic}'...")
        results = converter.generate_and_convert_podcast(
            args.topic,
            args.output,
            speakers,
            args.duration,
            args.voice_engine
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Generated podcast!")
            print(f"📁 Output: {results['output_dir']}")
            print(f"🎵 Audio files: {len(results['audio_files'])}")
            print(f"📝 Script saved: {results['script_file']}")
    
    elif args.command == "convert":
        if not args.file:
            print("❌ Script file required for convert mode")
            return
        
        print(f"📝 Converting script {args.file} to audio...")
        results = converter.convert_existing_script(
            args.file,
            args.output,
            voice_engine=args.voice_engine
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
        else:
            print(f"✅ Conversion complete!")
            print(f"📁 Output: {results['output_dir']}")
            print(f"🎵 Audio files: {len(results['audio_files'])}")
            print(f"👥 Speakers: {', '.join(results['speakers_found'])}")

if __name__ == "__main__":
    main()
