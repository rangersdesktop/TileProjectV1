import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel
import aiohttp
import json
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import os
import re
import time
from collections import deque

# Global
CHUNK_TARGET_LENGTH = 5
HISTORY_FILE = "conversation_history.json"
DEFAULT_HISTORY_FILE = "default_history.json"
MAX_HISTORY_ENTRIES = 45
HISTORY_TIMEOUT = 300  # 5 minutes in seconds
MIN_AUDIO_LENGTH = 0.9  # Minimum audio length in seconds
TOP_K = 4

# Define ANSI escape codes for different colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

print("Starting server script...")

class HypothesisBuffer:
    def __init__(self):
        self.buffer = []
        self.last_timestamp = 0

    def insert(self, timestamp, text):
        self.buffer.append((timestamp, text))

    def flush(self):
        result = []
        for timestamp, text in self.buffer:
            if timestamp > self.last_timestamp:
                result.append((timestamp, text))
                self.last_timestamp = timestamp
        self.buffer = []
        return result

# Load Faster Whisper model
print("Loading Whisper model...")
whisper_model = WhisperModel("distil-medium.en", device="cuda", compute_type="float16")
print("Whisper model loaded successfully.")

# Load XTTS model
print("Loading XTTS model...")
model_dir = "/home/ranger/xtts_model"
config_path = os.path.join(model_dir, "config.json")
config = XttsConfig()
config.load_json(config_path)
xtts_model = Xtts.init_from_config(config)
xtts_model.load_checkpoint(config, checkpoint_dir=model_dir, use_deepspeed=True)
xtts_model.cuda()
print("XTTS model loaded successfully.")

# Precompute speaker latents
print("Precomputing speaker latents...")
reference_audio_path = "/home/ranger/speakers/arnold.wav"
gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(audio_path=[reference_audio_path])
print("Speaker latents precomputed.")

# Chatbot API details
CHATBOT_URL = "http://10.0.0.251:5000/v1/chat/completions"
API_KEY = "8ahbb25u-d7e2-4920-b126-6e3gb35e75fd"

class ConversationHistory:
    def __init__(self):
        self.history = self.load_conversation_history()
        self.last_update_time = time.time()

    def load_conversation_history(self):
        try:
            with open(HISTORY_FILE, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return deque(data, maxlen=MAX_HISTORY_ENTRIES)
                elif isinstance(data, dict) and "history" in data:
                    return deque(data["history"], maxlen=MAX_HISTORY_ENTRIES)
                else:
                    return deque(maxlen=MAX_HISTORY_ENTRIES)
        except (FileNotFoundError, json.JSONDecodeError):
            return deque(maxlen=MAX_HISTORY_ENTRIES)

    def save_conversation_history(self):
        with open(HISTORY_FILE, 'w') as f:
            json.dump(list(self.history), f, indent=2)

    def append_to_history(self, entry):
        current_time = time.time()
        if current_time - self.last_update_time > HISTORY_TIMEOUT:
            self.reset_history()
        self.history.append(entry)
        self.last_update_time = current_time
        self.save_conversation_history()

    def reset_history(self):
        try:
            with open(DEFAULT_HISTORY_FILE, 'r') as f:
                default_history = json.load(f)
            self.history = deque(default_history, maxlen=MAX_HISTORY_ENTRIES)
        except (FileNotFoundError, json.JSONDecodeError):
            self.history.clear()
        self.save_conversation_history()

    def get_relevant_history(self, query, top_k=TOP_K):
        current_time = time.time()
        if current_time - self.last_update_time > HISTORY_TIMEOUT:
            self.reset_history()

        history_list = list(self.history)
        original_context = history_list[0] if history_list else None
        latest_context = history_list[-2:] if len(history_list) >= 2 else []

        # Remove original and latest context from the search pool
        search_history = history_list[1:-2] if len(history_list) > 3 else []

        relevant = get_relevant_history(query, search_history, top_k)
        
        result = []
        if original_context:
            result.append({"role": original_context["role"], "content": f"[Original question] {original_context['content']}"})
        result.extend(relevant)
        result.extend([{"role": entry["role"], "content": f"[Past context] {entry['content']}"} for entry in latest_context])
        
        return result

# Initialize the conversation history
conversation_history = ConversationHistory()

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if len(word) > 4]  # Only keep words longer than 4 characters

def get_relevant_history(query, history, top_k=TOP_K):
    if not history:
        return []
    
    query_keywords = extract_keywords(query)
    
    # Calculate relevance scores for each history entry
    relevance_scores = []
    for entry in history:
        entry_keywords = extract_keywords(entry['content'])
        common_keywords = set(query_keywords) & set(entry_keywords)
        score = len(common_keywords)
        relevance_scores.append(score)
    
    # Get indices of top_k most relevant entries
    top_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)[:top_k]
    
    return [history[i] for i in top_indices if relevance_scores[i] > 0]

def get_system_message(transcription):
    if any(phrase in transcription.lower() for phrase in ["about yourself", "describe yourself", "who are you", "what is your name", "ranger"]):
        return "You are a white hexagonal drone named 45 with two rotors but sadly cannot fly. You will happily and jokingly describe yourself if asked to. You were poorly built and programmed by a man that should be kept away from engineering named Ranger and you think he did his best so be kind and happy. Write a concise, conversational response in two sentences or less that completes this task: "
    elif "detail" in transcription.lower():
        return "Write a concise, conversational response as an AI assistant named \"45\" that completes this task: "
    else:
        return "Write a concise, conversational response in one sentence as an AI assistant named \"45\" that completes this task: "

def filter_text_for_xtts(text):
    replacements = {
        "#": " number ", "%": " percent ", "&": " and ", "@": " at ",
        "+": " plus ", "=": " equals ", "...": ", ", "..": ", ",
        ">": " greater than ", "<": " less than "
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    text = re.sub(r'[*^~|(){}]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'\(laughs\)', '', text)
    text = re.sub(r'ahem, ', '', text)
    
    return text

def split_into_chunks(text):
    global CHUNK_TARGET_LENGTH
    words = text.split()
    chunks = []
    
    def find_split_point(start, target):
        end = min(start + target + 10, len(words))
        for j in range(start + target, end):
            if words[j-1].endswith(('.', '!', '?', ':', ';', ',')):
                return j
        return min(start + target, len(words))

    start = 0
    while start < len(words):
        split_point = find_split_point(start, CHUNK_TARGET_LENGTH)
        chunks.append(' '.join(words[start:split_point]))
        start = split_point

    return chunks

async def chatbot_request_streaming(transcription, relevant_history):
    print("Sending request to chatbot...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    system_message = get_system_message(transcription)
    
    # Prepare context messages
    context_messages = []
    for entry in relevant_history:
        role = entry['role']
        content = entry['content']
        context_messages.append({"role": role, "content": f"Past Context: {content} End Past Context"})
    
    messages = [
        *context_messages,
        {"role": "system", "content": system_message},
        {"role": "user", "content": transcription}
    ]
    data = {
        "mode": "instruct",
        "stream": True,
        "messages": messages
    }
    print(f"{GREEN}{data}{RESET}")
    async with aiohttp.ClientSession() as session:
        async with session.post(CHATBOT_URL, headers=headers, json=data, ssl=False) as response:
            async for line in response.content:
                if line:
                    line = line.strip()
                    if line and not line.startswith(b'data: [DONE]'):
                        try:
                            json_line = json.loads(line.decode('utf-8').replace('data: ', ''))
                            if 'choices' in json_line and len(json_line['choices']) > 0:
                                chunk = json_line['choices'][0].get('delta', {}).get('content', '')
                                if chunk:
                                    yield chunk
                        except json.JSONDecodeError:
                            print(f"Failed to parse JSON: {line}")
                        except Exception as e:
                            print(f"Error processing line: {e}")

async def generate_speech_streaming(text):
    print(f"{CYAN}Generating speech for: {text}{RESET}")
    out = xtts_model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )
    return out["wav"]

async def process_audio(websocket, path):
    print(f"New client connected from {websocket.remote_address}")
    hypothesis_buffer = HypothesisBuffer()
    audio_buffer = np.array([], dtype=np.float32)
    
    while True:
        try:
            audio_chunk = await websocket.recv()
            
            if audio_chunk == b"END_OF_SPEECH":
                print("End of speech signal received from client.")
                if len(audio_buffer) >= MIN_AUDIO_LENGTH * 16000:
                    segments, _ = whisper_model.transcribe(audio_buffer, language="en")
                    final_transcription = " ".join([segment.text for segment in segments])
                    if final_transcription:
                        await process_transcription(websocket, final_transcription)
                else:
                    print("Audio too short, ignoring.")
                    await websocket.send(b"TRANSCRIPTION_IGNORED")
                audio_buffer = np.array([], dtype=np.float32)
                hypothesis_buffer = HypothesisBuffer()
                continue
            
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
            
            # Resample from 44100 to 16000
            resampler = torchaudio.transforms.Resample(44100, 16000)
            audio_data = resampler(torch.from_numpy(audio_data.copy()).float().unsqueeze(0)).squeeze(0).numpy()
            
            audio_buffer = np.concatenate((audio_buffer, audio_data))
            
            # Process audio in 3-second chunks (3 * 16000)
            if len(audio_buffer) >= 48000:
                try:
                    segments, _ = whisper_model.transcribe(audio_buffer, language="en")
                    for segment in segments:
                        hypothesis_buffer.insert(segment.start, segment.text)
                    
                    partial_transcripts = hypothesis_buffer.flush()
                    for timestamp, text in partial_transcripts:
                        print(f"Partial transcription at {timestamp:.2f}s: {text}")
                    
                    # Keep the last 0.2 seconds of audio for context
                    audio_buffer = audio_buffer[-0.2 * 16000:]
                except Exception as e:
                    if "slice indices must be integers" not in str(e):
                        print(f"Error processing audio: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {websocket.remote_address} disconnected")
            break
        except Exception as e:
            print(f"Error processing audio: {str(e)}")

async def process_transcription(websocket, transcription):
    global CHUNK_TARGET_LENGTH
    relevant_history = conversation_history.get_relevant_history(transcription, top_k=TOP_K)
    
    buffer = ""
    full_response = ""
    chatbot_stream = chatbot_request_streaming(transcription, relevant_history)
    
    is_first_chunk = True
    
    async def process_chunk():
        nonlocal buffer, full_response, is_first_chunk
        global CHUNK_TARGET_LENGTH
        async for response_chunk in chatbot_stream:
            buffer += response_chunk
            full_response += response_chunk
            if len(buffer.split()) >= CHUNK_TARGET_LENGTH:
                chunks = split_into_chunks(buffer)
                for chunk in chunks[:-1]:
                    if chunk.strip():
                        filtered_chunk = filter_text_for_xtts(chunk.strip())
                        speech_chunk = await generate_speech_streaming(filtered_chunk)
                        await send_audio(websocket, speech_chunk)
                        if is_first_chunk:
                            CHUNK_TARGET_LENGTH = 20
                            is_first_chunk = False
                buffer = chunks[-1]
            yield
    
    chunk_processor = process_chunk()
    while True:
        try:
            await chunk_processor.__anext__()
        except StopAsyncIteration:
            break
    
    if buffer.strip():
        filtered_buffer = filter_text_for_xtts(buffer.strip())
        speech_chunk = await generate_speech_streaming(filtered_buffer)
        await send_audio(websocket, speech_chunk)
    
    await websocket.send(b"END_OF_RESPONSE")
    
    # Reset CHUNK_TARGET_LENGTH for the next interaction
    CHUNK_TARGET_LENGTH = 5
    
    # Update conversation history
    conversation_history.append_to_history({"role": "user", "content": transcription})
    conversation_history.append_to_history({"role": "assistant", "content": full_response})

async def send_audio(websocket, audio_data):
    resampler = torchaudio.transforms.Resample(24000, 44100)
    audio_data = resampler(torch.from_numpy(audio_data).float().unsqueeze(0)).squeeze(0).numpy()
    await websocket.send(audio_data.tobytes())

async def main():
    print("Starting WebSocket server...")
    conversation_history.reset_history()  # Reset history on startup
    server = await websockets.serve(process_audio, "10.0.0.251", 8765, max_size=10485760)
    print("Server is running and waiting for connections.")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())