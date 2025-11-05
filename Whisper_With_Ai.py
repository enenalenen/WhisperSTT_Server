# -*- coding: utf-8 -*-
import subprocess
import os
import tempfile
import wave
import time
import re
import socket
import threading
import audioop
import sys
from thefuzz import fuzz

script_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(script_path)
print(f"[*] Added script path to sys.path: {script_path}")


import requests 
try:
    import keyboard
except ImportError:
    print("Error: 'keyboard' library not found. Please install it using: pip install keyboard")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("Error: 'numpy' library not found. Please install it using: pip install numpy")
    sys.exit(1)

import queue # 스레드 간 작업 분배
try:
    import torch
    import torchaudio
except ImportError as e:
    print(f"Error: Required AI libraries not found (torch, torchaudio).")
    print(f"Please install them using: pip install torch torchaudio")
    print(f"Details: {e}")
    sys.exit(1)


try:
    import preprocessor
except ImportError as e:
    print("\n[!!!] FATAL IMPORT ERROR: Failed to import 'preprocessor.py'.")
    print(f"    This likely means 'preprocessor.py' or one of its dependencies is missing or failed to load.")
    print(f"    Detailed Error: {e}")
    print("    Please ensure 'preprocessor.py' is in the same directory and all its required libraries (torch, torchaudio, numpy) are installed.")
    sys.exit(1)


# --- 설정 ---
HOST = '0.0.0.0'
AUDIO_PORT = 6789
COMMAND_PORT = 6790
SAVE_PATH = "C:/server/stt_server/recorded_audio"

# 오디오 스트림 설정
RATE = 16000 # 16kHz
CHANNELS = 1
SAMPWIDTH = 2
CHUNK = 2048 # (0.128초)


# VAD 설정 
NOISE_THRESHOLD = 2000 # 소리 인식 기준치
SILENCE_DURATION_SECONDS = 1.0
REQUIRED_SILENCE_CHUNKS = int(SILENCE_DURATION_SECONDS * RATE / CHUNK)
MIN_SPEECH_DURATION_SECONDS = 0.5 # (0.5초 미만 소음 무시)
REQUIRED_SPEECH_CHUNKS = int(MIN_SPEECH_DURATION_SECONDS * RATE / CHUNK) # (약 4개 청크)

# --- AI 모델 설정 ---

# 1. STT (Whisper) 설정
WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference" 
SIMILARITY_THRESHOLD = 85
TARGET_KEYWORDS = ["안녕", "감사", "조심", "위험"]
keywords_lock = threading.Lock()

# 2. SC (경보음 감지) 설정
SC_MODEL_PATH = "model/resnet50_best.pt" # (하위 'model' 폴더에 위치)

SC_SAMPLE_RATE = 16000 # (preprocessor.py와 일치시킴)
SC_NUM_SAMPLES = 16000 # (logmel 함수에 전달할 고정 1초 길이)
SC_N_FFT = 512
SC_HOP_LENGTH = 160
SC_N_MELS = 64


# CLASS_NAME = ["danger", "fire", "gas", "non", "tsunami"]
SC_CLASS_LABELS = {
    0: 'danger',  # -> "siren"으로 통합
    1: 'fire',    # -> "siren"으로 통합
    2: 'gas',     # -> "gas"로 출력
    3: 'non',     # -> 무시
    4: 'tsunami'  # -> "siren"으로 통합
}
# 처리할 인덱스 정의
SIREN_LIKE_INDICES = {0, 1, 4} # danger, fire, tsunami
GAS_INDEX = 2
NON_INDEX = 3


# --- 글로벌 변수 및 스레드 큐(Queue) ---
audio_client_socket = None
command_client_socket = None 
socket_lock = threading.Lock()
manual_recording_triggered = threading.Event()

# AI Worker용 작업 큐
stt_task_queue = queue.Queue(maxsize=10)     # STT (음성 버퍼) 작업 큐
sc_task_queue = queue.Queue(maxsize=30)      # SC (실시간 청크) 작업 큐
result_queue = queue.Queue(maxsize=10)       # 모든 AI의 최종 결과(문자열) 큐

try:
    print(f"[*] Loading Sound Classification model from {SC_MODEL_PATH}...")

    
    # PyTorch 2.6+ 호환을 위해 weights_only=False 추가
    sc_model = torch.load(SC_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False) 
    
    sc_model.eval() # 추론 모드로 설정
    print("[*] Sound Classification model loaded successfully.")
except FileNotFoundError:
    print(f"[!!!] FATAL ERROR: SC Model file not found at '{SC_MODEL_PATH}'")
    print("      Please ensure 'resnet50_best.pt' is placed in the 'model' subfolder.")
    sys.exit(1)
except Exception as e:
    print(f"[!!!] FATAL ERROR: Failed to load SC model: {e}")
    sys.exit(1)

# HighPassFilter 클래스 (오버플로우 버그 수정된 버전)
class HighPassFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.last_input = 0
        self.last_output = 0

    def process(self, input_data):
        samples = np.frombuffer(input_data, dtype=np.int16)
        filtered_samples = np.zeros_like(samples)
        
        last_in = np.float64(self.last_input)
        last_out = np.float64(self.last_output)
        alpha = self.alpha 

        INT16_MIN = -32768
        INT16_MAX = 32767

        for i in range(len(samples)):
            try:
                with np.errstate(over='raise', invalid='raise'):
                    last_out = alpha * (last_out + np.float64(samples[i]) - last_in)
                    last_in = np.float64(samples[i])
                if not np.isfinite(last_out):
                    last_in = 0.0
                    last_out = 0.0
                if last_out > INT16_MAX:
                    last_out = np.float64(INT16_MAX)
                elif last_out < INT16_MIN:
                    last_out = np.float64(INT16_MIN)
            except (FloatingPointError, OverflowError):
                last_in = 0.0
                last_out = 0.0
            filtered_samples[i] = int(last_out)
            
        self.last_input = int(last_in)
        self.last_output = int(last_out)
        return filtered_samples.tobytes()

# create_wav_file 함수
def create_wav_file(pcm_data, rate, is_temp=True, filename_prefix="rec_"):
    timestamp = int(time.time())
    if is_temp:
        dir_path = tempfile.gettempdir()
        filename = f"{filename_prefix}{timestamp}.wav"
    else:
        dir_path = SAVE_PATH
        filename = f"{filename_prefix}{timestamp}.wav"
    file_path = os.path.join(dir_path, filename)
    file_path = os.path.normpath(file_path)
    if not is_temp:
        print(f"\nCreating WAV file at: {file_path}")
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPWIDTH)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return file_path

# run_stt 함수 (환각 억제 파라미터 적용된 버전)
def run_stt(file_path):
    normalized_path = os.path.normpath(file_path)
    print(f"\n(STT Worker) Running STT on {normalized_path} via HTTP...")
    try:
        with open(normalized_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            data = {
                'response_format': 'json', 'language': 'ko',
                'temperature': 0.0,
                'logprob_threshold': -0.5, 
                'no_speech_threshold': 0.7,
                'prompt': ''
            }
            response = requests.post(WHISPER_SERVER_URL, files=files, data=data)
            if response.status_code == 200:
                result_json = response.json()
                transcribed_text = result_json.get('text', '')
                cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', transcribed_text).strip()
                print(f"(STT Worker) Transcription Result: {cleaned_text}")
                return cleaned_text
            else:
                print(f"(STT Worker) Whisper server returned error: {response.status_code}")
                return None
    except requests.exceptions.RequestException as e:
        print(f"(STT Worker) Failed to connect to Whisper server: {e}")
        return None
    except Exception as e:
        print(f"(STT Worker) An error occurred during STT request: {e}")
        return None
    finally:
        if tempfile.gettempdir() in normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)

# AI Worker 1: STT (음성 인식)
def stt_worker():
    """stt_task_queue에서 VAD가 완료된 오디오 버퍼를 받아 STT를 수행"""
    print("[*] STT Worker thread started, waiting for speech tasks...")
    while True:
        try:
            # 큐에 작업이 올 때까지 대기 (bytes)
            audio_buffer = stt_task_queue.get() 
            if audio_buffer is None: break # 스레드 종료 신호
            
            temp_wav_path = create_wav_file(bytes(audio_buffer), RATE, is_temp=True)
            text_result = run_stt(temp_wav_path)
            
            if text_result: # STT 결과가 비어있지 않다면
                keywords_to_send = []
                with keywords_lock: current_target_keywords = TARGET_KEYWORDS[:]
                
                # 1. 사용자 지정 키워드 검사 (유사도)
                for keyword in current_target_keywords:
                    if fuzz.partial_ratio(keyword, text_result) >= SIMILARITY_THRESHOLD:
                        keywords_to_send.append(keyword)
                
                # 2. 경보음 키워드 검사 (STT 결과에도 경보음 단어가 포함될 수 있음)
                for alarm_keyword in ["siren", "horn", "boom", "explosion"]:
                    if alarm_keyword in text_result.lower():
                        if "siren" in alarm_keyword: keywords_to_send.append("siren")
                        elif "horn" in alarm_keyword: keywords_to_send.append("horn")
                        elif "boom" in alarm_keyword: keywords_to_send.append("boom")

                if keywords_to_send:
                    response_message = ",".join(sorted(list(set(keywords_to_send))))
                    # 소켓 대신 result_queue로 결과 전송
                    result_queue.put(response_message)
                    print(f"(STT Worker) Queued result: {response_message}")
        
        except Exception as e:
            print(f"[!!!] STT Worker error: {e}")
            time.sleep(1) # 오류 발생 시 잠시 대기

# AI Worker 2: SC (경보음 감지) - '무음' 방어 코드 추가
def sc_worker():
    """sc_task_queue에서 실시간 청크를 받아 1초 윈도우로 경보음을 감지"""
    print("[*] SC Worker thread started, waiting for audio chunks...")
    
    # 1초(16000 샘플) 버퍼. 0.5초(8000 샘플)마다 분석
    sc_audio_buffer = np.array([], dtype=np.int16)
    SLIDING_STEP_SAMPLES = 8000 # 0.5초

    # 스팸 방지용 변수
    SIREN_CHUNK_THRESHOLD = 3 # 3연속 'siren' 계열 감지 시 알림
    GAS_CHUNK_THRESHOLD = 3   # 3연속 'gas' 계열 감지 시 알림
    ALERT_COOLDOWN_SECONDS = 10 # 최소 10초에 한 번만 알림
    
    siren_counter = 0
    gas_counter = 0
    last_siren_alert_time = 0
    last_gas_alert_time = 0

    # 경고 '종료' 감지를 위한 상태 변수
    siren_active_state = False
    gas_active_state = False
    siren_silence_counter = 0
    gas_silence_counter = 0
    
    # 0.5초(1 청크) * 5 = 2.5초 동안 'non'이 지속되면 종료로 간주
    SILENCE_STOP_THRESHOLD = 5 
    GAS_STOP_THRESHOLD = 5
    
    # '순수 디지털 무음' 오탐지 방지용 임계값
    # (int16 최대값 32767 기준, 50 이하는 무시)
    MIN_AUDIO_THRESHOLD = 50 

    while True:
        try:
            # 큐에 작업이 올 때까지 대기 (numpy array)
            audio_chunk_np = sc_task_queue.get() 
            if audio_chunk_np is None: break # 스레드 종료 신호

            # 1. 내부 버퍼에 오디오 청크 추가
            sc_audio_buffer = np.concatenate([sc_audio_buffer, audio_chunk_np])

            # 2. 버퍼가 1초(SC_NUM_SAMPLES) 이상 쌓였는지 확인
            while len(sc_audio_buffer) >= SC_NUM_SAMPLES:
                # 2a. 최신 1초 분량의 오디오 클립 추출
                audio_clip_to_analyze = sc_audio_buffer[-SC_NUM_SAMPLES:]

                # '순수 디지털 무음'(all-zeros)이 'siren'으로 오탐지되는 문제 해결
                if np.abs(audio_clip_to_analyze).max() < MIN_AUDIO_THRESHOLD:
                    predicted_class = NON_INDEX # 3 (='non')
                    print(f"(SC Worker) Pure silence chunk detected (Max: {np.abs(audio_clip_to_analyze).max()}), forcing 'non'.")
                else:
                    # 2b. preprocessor.logmel 함수 호출
                    mel_spec_np = preprocessor.logmel(
                        audio_clip_to_analyze, 
                        num_samples=SC_NUM_SAMPLES,
                        sample_rate=SC_SAMPLE_RATE,
                        n_fft=SC_N_FFT,
                        hop_length=SC_HOP_LENGTH,
                        n_mels=SC_N_MELS
                    )
                    
                    # preprocessor.logmel이 (B, C, H, W) 형태의 numpy를 반환
                    mel_spec_tensor = torch.from_numpy(mel_spec_np)

                    # 2c. 추론 (신규 AI 모델)
                    with torch.no_grad():
                        output = sc_model(mel_spec_tensor)
                        _, predicted_idx = torch.max(output.data, 1)
                    
                    predicted_class = predicted_idx.item()


                # 2d. 결과 판정 (사용자 요청 로직)
                current_time = time.time()
                
                # '시작' 카운터와 '종료' 카운터를 분리하여 관리
                
                if predicted_class in SIREN_LIKE_INDICES: # (0, 1, 4)
                    siren_counter += 1
                    gas_counter = 0 
                    siren_silence_counter = 0 # 'non' 카운터 초기화
                    print(f"(SC Worker) 'siren-like' chunk detected! (Count: {siren_counter})")
                    
                elif predicted_class == GAS_INDEX: # (2)
                    gas_counter += 1
                    siren_counter = 0
                    gas_silence_counter = 0 # 'non' 카운터 초기화
                    print(f"(SC Worker) 'gas' chunk detected! (Count: {gas_counter})")
                    
                else: # (3, 'non')
                    siren_counter = 0 # '시작' 카운터 초기화
                    gas_counter = 0   # '시작' 카운터 초기화
                    
                    # 경고가 활성화된 상태였다면, '종료' 카운터 증가
                    if siren_active_state:
                        siren_silence_counter += 1
                        print(f"(SC Worker) 'non' chunk while siren active (Silence Count: {siren_silence_counter})")
                    if gas_active_state:
                        gas_silence_counter += 1
                        print(f"(SC Worker) 'non' chunk while gas active (Silence Count: {gas_silence_counter})")

                # 2e. 스팸 방지 및 '시작'/'종료' 결과 전송
                
                # 1. 사이렌 "시작" 로직
                if siren_counter >= SIREN_CHUNK_THRESHOLD and (current_time - last_siren_alert_time > ALERT_COOLDOWN_SECONDS):
                    result_queue.put("siren") 
                    print(f"(SC Worker) Queued result: siren")
                    siren_counter = 0 # 쿨다운을 위해 카운터 리셋
                    last_siren_alert_time = current_time 
                    siren_active_state = True  # 상태 활성화
                    siren_silence_counter = 0  # 'non' 카운터 초기화
                
                # 2. 가스 "시작" 로직
                if gas_counter >= GAS_CHUNK_THRESHOLD and (current_time - last_gas_alert_time > ALERT_COOLDOWN_SECONDS):
                    result_queue.put("gas") 
                    print(f"(SC Worker) Queued result: gas")
                    gas_counter = 0 
                    last_gas_alert_time = current_time 
                    gas_active_state = True  # 상태 활성화
                    gas_silence_counter = 0  # 'non' 카운터 초기화
                
                # 3. 사이렌 "종료" 로직
                if siren_active_state and siren_silence_counter >= SILENCE_STOP_THRESHOLD:
                    result_queue.put("siren_stopped")
                    print(f"(SC Worker) Queued result: siren_stopped")
                    siren_active_state = False # 상태 비활성화
                    siren_silence_counter = 0  # 'non' 카운터 초기화
                
                # 4. 가스 "종료" 로직
                if gas_active_state and gas_silence_counter >= GAS_STOP_THRESHOLD:
                    result_queue.put("gas_stopped")
                    print(f"(SC Worker) Queued result: gas_stopped")
                    gas_active_state = False # 상태 비활성화
                    gas_silence_counter = 0  # 'non' 카운터 초기화

                # 2f. 슬라이딩 윈도우: 0.5초(SLIDING_STEP_SAMPLES) 분량의 오래된 데이터 삭제
                sc_audio_buffer = sc_audio_buffer[SLIDING_STEP_SAMPLES:]

        except Exception as e:
            print(f"[!!!] SC Worker error: {e}")
            sc_audio_buffer = np.array([], dtype=np.int16) # 오류 시 버퍼 초기화
            # 큐에 쌓인 오래된 데이터 비우기 (오류 복구)
            while not sc_task_queue.empty():
                try: sc_task_queue.get_nowait()
                except queue.Empty: break
            time.sleep(1)


# handle_audio_client (오디오 생산자)
def handle_audio_client(client_socket):
    global audio_client_socket
    with socket_lock: audio_client_socket = client_socket
    
    print("[*] Audio client connected. Starting audio processing pipeline...")
    print(">>> Press 'g' to start a 5-second manual recording <<<")
    
    hp_filter = HighPassFilter()
    
    # STT VAD용 변수
    audio_buffer = bytearray()
    is_speaking = False
    silence_counter = 0
    speech_chunk_counter = 0
    
    # 녹음용 변수
    manual_recording_buffer = bytearray()
    manual_recording_start_time = 0

    try:
        while True:
            data = client_socket.recv(CHUNK * SAMPWIDTH)
            if not data: break
            
            # 1. 하이패스 필터 적용
            filtered_data_bytes = hp_filter.process(data)
            
            # --- 2. SC Worker로 실시간 전송 ---
            try:
                # SC Worker는 NumPy 배열을 사용
                audio_chunk_np = np.frombuffer(filtered_data_bytes, dtype=np.int16)
                # 큐가 꽉 찼으면(non-blocking) 무시 (실시간성 유지)
                sc_task_queue.put_nowait(audio_chunk_np) 
            except queue.Full:
                pass # SC 처리가 밀리면 이 청크는 건너뜀
            except Exception as e:
                print(f"[!] SC Queue Error: {e}")

            # --- 3. STT Worker를 위한 VAD 로직 ---
            rms = audioop.rms(filtered_data_bytes, SAMPWIDTH)
            print(f"RMS: {rms:<5}\r", end="")
            sys.stdout.flush()

            # 녹음 로직
            if manual_recording_triggered.is_set():
                if manual_recording_start_time == 0:
                    print("\n*** Starting 5-second manual recording... ***")
                    manual_recording_start_time = time.time()
                manual_recording_buffer.extend(filtered_data_bytes)
                if time.time() - manual_recording_start_time >= 5:
                    print("\n*** Finished manual recording. Saving file... ***")
                    create_wav_file(manual_recording_buffer, RATE, is_temp=False, filename_prefix="manual_rec_")
                    manual_recording_buffer.clear(); manual_recording_start_time = 0
                    manual_recording_triggered.clear()
                    print(">>> Press 'g' to start a new 5-second recording <<<")

            # VAD 로직
            if rms > NOISE_THRESHOLD:
                if not is_speaking: print("\n(VAD) Speaking detected...")
                is_speaking = True
                silence_counter = 0
                audio_buffer.extend(filtered_data_bytes)
                speech_chunk_counter += 1
                
            elif is_speaking:
                silence_counter += 1
                audio_buffer.extend(filtered_data_bytes)
                
                if silence_counter > REQUIRED_SILENCE_CHUNKS:
                    # STT 요청 전, 최소 음성 길이 확인
                    if speech_chunk_counter >= REQUIRED_SPEECH_CHUNKS:
                        print(f"\n(VAD) Speech long enough ({speech_chunk_counter} chunks), queueing for STT...")
                        # STT 함수 직접 호출 대신, stt_task_queue에 버퍼(bytes)를 넣음
                        try:
                            stt_task_queue.put_nowait(bytes(audio_buffer))
                        except queue.Full:
                            print("(VAD) STT task queue is full, dropping this speech.")
                    else:
                        print(f"\n(VAD) Speech too short ({speech_chunk_counter} < {REQUIRED_SPEECH_CHUNKS}), ignoring as noise burst...")
                    
                    # 버퍼 및 상태 변수 초기화
                    audio_buffer.clear()
                    is_speaking = False
                    silence_counter = 0
                    speech_chunk_counter = 0

    except (ConnectionResetError, socket.error):
        print("\n[*] Audio client connection lost.")
    finally:
        with socket_lock: audio_client_socket = None
        print("\n[*] Audio client connection closed.")


# handle_command_client (명령 수신 및 결과 송신)
def handle_command_client(client_socket):
    global TARGET_KEYWORDS, NOISE_THRESHOLD, command_client_socket
    print("[*] Command client connected.")
    with socket_lock:
        command_client_socket = client_socket # 전역 변수에 소켓 저장
    
    # --- 결과(STT, SC)를 앱으로 전송하는 스레드 ---
    def result_sender(sock):
        """result_queue에서 결과를 받아 앱으로 전송"""
        print("[*] Result sender thread started, waiting for AI results...")
        try:
            while True:
                # 큐에 결과(예: "조심" 또는 "siren" 또는 "siren_stopped")가 올 때까지 대기
                result_message = result_queue.get() 
                if result_message is None: break # 스레드 종료 신호
                
                if sock and sock.fileno() != -1:
                    print(f"\n[!!!] >>> Sending Result to App: {result_message} <<<\n")
                    sock.sendall((result_message + '\n').encode('utf-8'))
                else:
                    print("[*] Result sender: Socket closed, stopping thread.")
                    break # 소켓이 닫히면 중지
        except (socket.error, IOError) as e:
            print(f"[*] Result sender thread error: {e}")
        except Exception as e:
            print(f"[!!!] Result sender thread fatal error: {e}")

    sender_thread = threading.Thread(target=result_sender, args=(client_socket,))
    sender_thread.daemon = True
    sender_thread.start()
    # --- 전송 스레드 끝 ---

    try:
        # --- 앱으로부터 명령(키워드, 민감도)을 수신하는 로직 ---
        socket_file = client_socket.makefile('r', encoding='utf-8')
        for line in socket_file:
            command = line.strip()
            if not command:
                continue
            print(f"[*] Received command from app: {command}")
            
            if command.startswith("CMD_UPDATE_KEYWORDS:"):
                new_keywords_str = command.split(':', 1)[1]
                with keywords_lock: 
                    TARGET_KEYWORDS = [kw.strip() for kw in new_keywords_str.split(',') if kw.strip()]
                print(f"\n[!!!] TARGET_KEYWORDS updated by app: {TARGET_KEYWORDS}\n")
            
            elif command.startswith("CMD_SET_SENSITIVITY:"):
                try:
                    app_value = int(command.split(':', 1)[1])
                    app_value = max(1, min(10, app_value)) 
                    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                    # [수정] VAD 민감도 매핑 범위 변경
                    MIN_NOISE = 4000 # 둔감 (기존 2000)
                    MAX_NOISE = 400  # 민감 (기존 200)
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                    mapped_threshold = MIN_NOISE + ((app_value - 1) / 9.0) * (MAX_NOISE - MIN_NOISE)
                    NOISE_THRESHOLD = int(mapped_threshold) 
                    print(f"\n[!!!] NOISE SENSITIVITY updated by app: {app_value} (Threshold: {NOISE_THRESHOLD})\n")
                except Exception as e:
                    print(f"[!] Failed to parse sensitivity command: {e}")
        
    except (ConnectionResetError, socket.error):
         print("[*] Command client connection lost (Receiver).")
    except Exception as e:
        print(f"[!!!] Command client fatal error: {e}")
    finally:
        print("[*] Command client connection closed.")
        with socket_lock:
            command_client_socket = None # 소켓 연결 종료 시 전역 변수 초기화
        # result_sender 스레드가 큐에서 계속 기다리지 않도록 종료 신호 전송
        result_queue.put(None)


# start_server 함수
def start_server(port, handler):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, port))
        server_socket.listen()
        print(f"[*] Server listening on {HOST}:{port}")
        while True:
            client_socket, addr = server_socket.accept()
            print(f"\n[*] Accepted connection from {addr} on port {port}")
            client_handler = threading.Thread(target=handler, args=(client_socket,))
            client_handler.daemon = True
            client_handler.start()

# keyboard_listener 함수
def keyboard_listener():
    print("(Keyboard listener started)")
    keyboard.add_hotkey('g', lambda: manual_recording_triggered.set())
    while True:
        time.sleep(1)

# ▼▼▼ main 함수 (AI Worker 스레드 시작) ▼▼▼
def main():
    # 1. Whisper 서버 헬스 체크
    try:
        response = requests.get(WHISPER_SERVER_URL.replace("/inference", "/health"))
        if response.status_code != 200 or response.json().get("status") != "ok":
             raise Exception("Server not healthy")
        print(f"[*] Connected to whisper.cpp server at {WHISPER_SERVER_URL}")
    except Exception as e:
        print(f"[!!!] FATAL ERROR: Cannot connect to whisper.cpp server at {WHISPER_SERVER_URL}")
        print("      Please ensure 'whisper-server.exe' is running (with -l ko) on port 8081.")
        sys.exit(1)

    # 2. 녹음 폴더 생성
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    
    # 3. 키보드 리스너 시작
    k_thread = threading.Thread(target=keyboard_listener)
    k_thread.daemon = True
    k_thread.start()
    
    # 4. AI Worker 스레드 시작
    print("[*] Starting AI Workers...")
    stt_thread = threading.Thread(target=stt_worker)
    stt_thread.daemon = True
    stt_thread.start()
    
    sc_thread = threading.Thread(target=sc_worker)
    sc_thread.daemon = True
    sc_thread.start()
    print("[*] AI Workers started.")

    # 5. TCP 서버 스레드 시작
    audio_thread = threading.Thread(target=start_server, args=(AUDIO_PORT, handle_audio_client))
    command_thread = threading.Thread(target=start_server, args=(COMMAND_PORT, handle_command_client))
    audio_thread.daemon = True
    command_thread.daemon = True
    audio_thread.start()
    command_thread.start()
    
    # 6. 메인 스레드 대기
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Server is shutting down.")
        # 종료 시 워커 스레드들도 정리
        stt_task_queue.put(None)
        sc_task_queue.put(None)
        result_queue.put(None)

if __name__ == "__main__":
    main()