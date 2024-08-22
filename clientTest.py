import asyncio
import websockets
import sounddevice as sd
import numpy as np
import time

SERVER_ADDRESS = "ws://10.0.0.251:8765"
SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = 'float32'
CHUNK_DURATION = 0.2
NOISE_FLOOR_DURATION = 1
WAKEUP_THRESHOLD = 2.7
SPEECH_END_THRESHOLD = 1.4
SPEECH_END_DURATION = 0.4
#NOISE_UPDATE_INTERVAL = 30
RESPONSE_TIMEOUT = 10
POST_RESPONSE_DELAY = 0.2
WAKE_TIMEOUT = 2
KEEP_ALIVE_INTERVAL = 20

async def record_and_stream():
    while True:
        try:
            print("Connecting to server...")
            async with websockets.connect(
                SERVER_ADDRESS,
                max_size=None,
                ping_interval=30,
                ping_timeout=60,
                close_timeout=10
            ) as websocket:
                print("Connected to server.")
                
                keep_alive_task = asyncio.create_task(keep_alive(websocket))
                
                stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, device=1)
                with stream:
                    noise_floor = await measure_noise_floor(stream)
                    print(f"Initial noise floor: {noise_floor}")
                    last_wakeup_time = time.time()
                    
                    while True:
                        try:
                            await asyncio.wait_for(websocket.ping(), timeout=5)
                            
                            await wait_for_wakeup(stream, noise_floor, websocket)
                            last_wakeup_time = time.time()
                            if await stream_conversation(stream, websocket, noise_floor):
                                await asyncio.sleep(POST_RESPONSE_DELAY)
                                if not await check_for_follow_up(stream, noise_floor, websocket):
                                    print("No follow-up detected. Waiting for wake-up phrase...")
                            
                            #if time.time() - last_wakeup_time > NOISE_UPDATE_INTERVAL:
                            #    noise_floor = await update_noise_floor(stream, noise_floor)
                            #    last_wakeup_time = time.time()
                            
                            await websocket.ping()

                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            print("Connection lost. Reconnecting...")
                            break
                        
                        await asyncio.sleep(1)
                
                keep_alive_task.cancel()
        
        except Exception as e:
            print(f"An error occurred: {str(e)}. Reconnecting...")
        
        await asyncio.sleep(1)

async def keep_alive(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
        except:
            break

async def wait_for_wakeup(stream, noise_floor, websocket):
    print("Waiting for wake-up phrase...")
    while True:
        audio_chunk, _ = stream.read(int(CHUNK_DURATION * SAMPLE_RATE))
        energy = np.mean(np.abs(audio_chunk))
        if energy > noise_floor * WAKEUP_THRESHOLD:
            print("Wake-up detected!")
            return

async def stream_conversation(stream, websocket, noise_floor):
    print("Streaming conversation...")
    low_energy_start = None
    conversation_start = time.time()
    audio_sent = False
    end_of_prompt_time = None
    
    while True:
        audio_chunk, _ = stream.read(int(CHUNK_DURATION * SAMPLE_RATE))
        energy = np.mean(np.abs(audio_chunk))
        
        if energy > noise_floor * SPEECH_END_THRESHOLD:
            low_energy_start = None
            await websocket.send(audio_chunk.tobytes())
            audio_sent = True
        else:
            if low_energy_start is None:
                low_energy_start = time.time()
            elif time.time() - low_energy_start > SPEECH_END_DURATION:
                if audio_sent:
                    print("End of speech detected.")
                    end_of_prompt_time = time.time()
                    await websocket.send(b"END_OF_SPEECH")
                    break
                else:
                    print("No speech detected. Returning to wait state.")
                    await websocket.send(b"CANCEL_STREAM")
                    return False
            await websocket.send(audio_chunk.tobytes())
        
        if time.time() - conversation_start > WAKE_TIMEOUT and not audio_sent:
            print("False wake-up detected. Returning to wait state.")
            await websocket.send(b"CANCEL_STREAM")
            return False
    
    print("Waiting for response...")
    response_start_time = time.time()
    first_audio_received = False
    
    try:
        while True:
            response = await asyncio.wait_for(websocket.recv(), timeout=RESPONSE_TIMEOUT)
            if response == b"END_OF_RESPONSE":
                break
            elif response == b"TRANSCRIPTION_IGNORED":
                print("Transcription ignored due to short duration.")
                return False
            
            if not first_audio_received:
                first_audio_received = True
                latency = time.time() - end_of_prompt_time
                print(f"Response latency: {latency:.3f} seconds")
            
            audio_array = np.frombuffer(response, dtype=DTYPE)
            sd.play(audio_array, SAMPLE_RATE)
            sd.wait()
            
            response_start_time = time.time()
    
    except asyncio.TimeoutError:
        print("Timeout waiting for server response. Returning to wait state.")
        return False
    except Exception as e:
        print(f"Error receiving response: {str(e)}")
        return False
    
    print("Response finished. Ready for next input.")
    return True

async def check_for_follow_up(stream, noise_floor, websocket):
    print("Checking for follow-up...")
    follow_up_start = time.time()
    while time.time() - follow_up_start < WAKE_TIMEOUT:
        audio_chunk, _ = stream.read(int(CHUNK_DURATION * SAMPLE_RATE))
        energy = np.mean(np.abs(audio_chunk))
        if energy > noise_floor * WAKEUP_THRESHOLD:
            print("Follow-up detected!")
            return await stream_conversation(stream, websocket, noise_floor)
    return False

async def measure_noise_floor(stream):
    print(f"Measuring noise floor for {NOISE_FLOOR_DURATION} seconds...")
    noise_samples = []
    for _ in range(int(NOISE_FLOOR_DURATION / CHUNK_DURATION)):
        audio_chunk, _ = stream.read(int(CHUNK_DURATION * SAMPLE_RATE))
        noise_samples.append(np.mean(np.abs(audio_chunk)))
    return np.mean(noise_samples)

async def update_noise_floor(stream, current_noise_floor):
    print("Updating noise floor...")
    new_sample = await measure_noise_floor(stream)
    updated_noise_floor = (current_noise_floor * 4 + new_sample) / 5
    print(f"Updated noise floor: {updated_noise_floor}")
    return updated_noise_floor

print("Starting client...")
while True:
    try:
        asyncio.get_event_loop().run_until_complete(record_and_stream())
    except KeyboardInterrupt:
        print("Client stopped by user.")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}. Restarting the client...")
        time.sleep(1)