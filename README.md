# The client and server scripts for my Tile V1 project
Low latency AI Voice Chat on a local network (FasterWhisper > Any Chatbot API > XTTS).

**Features:**
It uses a RAG history system with optional default history for things you want your AI to always remember, dynamic keyword based system prompts, Volume based "wakeup" of the client, Allows for instant response after the chatbot has finished to create a more natural conversation, adjustable "chunk target length" for XTTS to ensure the first chunk is always fast but the rest are longer for quality

**My attempt at low latency:**
Audio is streamed from the client and transcribed in chunks by whisper with a sliding buffer. Full transcription is sent to the AI model along with the first and latest prompt (still working this out) and relevant context from the RAG search. This output is streamed into XTTS in chunks with the first chunk targeting 5 words and the rest targeting 20. The splitting function always tries to break at punctuation or commas. The audio is streamed back to the client and the latency from when the prompt ends to when the audio starts is printed.

Latency for this version with context tends to be around 1.5-2 sec but I'll keep working at it. (Also whisper doesn't have a warm up yet so the first prompt is slower)

**[This is all very much a work in progress! Check my website for details about the project https://rangersdesk.top]**

# **For "serverTest.py":**
Tested on Ubuntu 24.04 LTS
**Make sure you have Cuda setup before starting:**
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --break-system-packages

install text-generation-webui and download a model first:

https://github.com/oobabooga/text-generation-webui

My current favorite: "cd text-generation-webui/ && python3 download-model.py LoneStriker/Hermes-3-Llama-3.1-8B-6.0bpw-h6-exl2"

sudo apt-get update

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt update

sudo apt install python3.11 python3.11-venv python3.11-dev

python3.11 -m venv tts-env

source tts-env/bin/activate

pip3 install asyncio websockets numpy aiohttp torch torchaudio faster-whisper TTS

sudo apt-get install libsndfile1 libportaudio2 libportaudiocpp0

NOTE: This current version allows you to clone a voice using XTTS. Make sure to download the XTTSv2 model and set the config and voice sample paths

Run the server script:
source tts-env/bin/activate && python3 serverTest.py



# **For "clientTest.py":**
Can be run on anything with a microphone. I'm using a raspberry pi but I'll add support for an esp32 later.

sudo apt-get update

sudo apt-get install python3 python3-pip libportaudio2

pip3 install asyncio websockets sounddevice numpy

python3 -m sounddevice

You might need to adjust the specified "device = 1" for audio in the client script.

Run the client script:
python3 clientTest.py
