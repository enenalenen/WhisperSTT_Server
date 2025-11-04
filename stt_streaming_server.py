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

# ▼▼▼ RealtimeSTT 대신 requests를 다시 사용합니다 ▼▼▼
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
# ▲▲▲ 수정 끝 ▲▲▲


# --- 설정 ---
HOST = '0.0.0.0'
AUDIO_PORT = 6789
COMMAND_PORT = 6790
SAVE_PATH = "C:/server/stt_server/recorded_audio"

# 오디오 스트림 설정
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 샘플 레이트를 16000으로 수정합니다.
RATE = 16000
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
CHANNELS = 1
SAMPWIDTH = 2
CHUNK = 2048

# VAD 설정 (앱의 기본값 5에 해당하는 300으로 설정, 범위 50-500)
NOISE_THRESHOLD = 300 # 앱에서 변경 가능한 전역 변수
SILENCE_DURATION_SECONDS = 1.0
REQUIRED_SILENCE_CHUNKS = int(SILENCE_DURATION_SECONDS * RATE / CHUNK)

# ▼▼▼ [버그 수정 5] 사용자가 제안한 '최소 음성 길이' 필터 설정 ▼▼▼
# (0.5초 / (2048/16000)초) -> 0.5 / 0.128 = 3.9 (약 4개 청크)
MIN_SPEECH_DURATION_SECONDS = 0.5 
REQUIRED_SPEECH_CHUNKS = int(MIN_SPEECH_DURATION_SECONDS * RATE / CHUNK)
# ▲▲▲ [버그 수정 5 완료] ▲▲▲

# ▼▼▼ whisper-server.exe 주소를 다시 사용합니다 ▼▼▼
WHISPER_SERVER_URL = "http://127.0.0.1:8081/inference" 
# ▲▲▲ 수정 끝 ▲▲▲

# 키워드 설정
SIMILARITY_THRESHOLD = 85
TARGET_KEYWORDS = ["안녕", "감사", "조심", "위험"]
keywords_lock = threading.Lock()
ALARM_SOUND_KEYWORDS = ["siren", "horn", "boom", "explosion"]

# (문제 2 해결됨)
audio_client_socket = None
command_client_socket = None # STT 결과를 보낼 명령어 소켓
socket_lock = threading.Lock()
manual_recording_triggered = threading.Event()

class HighPassFilter:
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.last_input = 0
        self.last_output = 0

    # ▼▼▼ [버그 수정 1] 오버플로우 방지 및 상태 초기화 로직이 적용된 process 함수 ▼▼▼
    def process(self, input_data):
        samples = np.frombuffer(input_data, dtype=np.int16)
        filtered_samples = np.zeros_like(samples)
        
        # 상태 변수를 64비트 부동소수점(float64)으로 로드하여 계산 정밀도 향상
        last_in = np.float64(self.last_input)
        last_out = np.float64(self.last_output)
        alpha = self.alpha # (alpha는 이미 float)

        INT16_MIN = -32768
        INT16_MAX = 32767

        for i in range(len(samples)):
            try:
                # 계산 중 오버플로우가 발생하면 RuntimeWarning 대신 예외(Error)를 발생시킴
                with np.errstate(over='raise', invalid='raise'):
                    
                    # 64비트 float으로 계산하여 중간 과정의 오버플로우 방지
                    last_out = alpha * (last_out + np.float64(samples[i]) - last_in)
                    last_in = np.float64(samples[i])

                # 계산 결과가 비정상적인 값(inf: 무한대, NaN: 숫자 아님)인지 확인
                if not np.isfinite(last_out):
                    # 비정상 값이면 필터 상태를 0으로 강제 초기화 (오염 방지)
                    last_in = 0.0
                    last_out = 0.0
                
                # 오디오 샘플은 16비트 정수 범위를 넘지 않도록 값 제한 (클리핑)
                if last_out > INT16_MAX:
                    last_out = np.float64(INT16_MAX)
                elif last_out < INT16_MIN:
                    last_out = np.float64(INT16_MIN)

            except (FloatingPointError, OverflowError):
                # 'with np.errstate' 구문에서 예외가 발생한 경우 (오버플로우 감지)
                # 필터 상태를 즉시 0으로 초기화하여 다음 샘플부터 정상 처리되도록 함
                last_in = 0.0
                last_out = 0.0
                
            filtered_samples[i] = int(last_out)
            
        # 다시 int형으로 변환하여 다음 처리를 위해 상태 저장
        self.last_input = int(last_in)
        self.last_output = int(last_out)
        
        return filtered_samples.tobytes()
    # ▲▲▲ [버그 수정 1 완료] ▲▲▲

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

# ▼▼▼ run_stt 함수 (requests 사용) ▼▼▼
def run_stt(file_path):
    """
    whisper.cpp 서버(server.exe)에 HTTP POST 요청을 보내 STT를 수행합니다.
    """
    normalized_path = os.path.normpath(file_path)
    print(f"\nRunning STT on {normalized_path} via HTTP...")

    try:
        with open(normalized_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            
            # ▼▼▼ [버그 수정 4] 환각 억제 파라미터를 더 공격적으로 수정 ▼▼▼
            data = {
                'response_format': 'json',
                'language': 'ko',
                
                # 1. 확정적 디코딩 (수정 없음)
                'temperature': 0.0,
                
                # 2. 로그 확률 임계값: -1.0 -> -0.5로 상향. (환각 억제 강화)
                #    모델이 -0.5보다 낮은 확률로 추측하는 단어는 무시함.
                'logprob_threshold': -0.5, 
                
                # 3. '음성 없음' 임계값: 0.6 -> 0.7로 상향. (노이즈 억제 강화)
                #    모델이 70% 이상 '음성 없음'으로 판단하면 빈 텍스트 반환
                'no_speech_threshold': 0.7,
                
                # 4. 프롬프트 초기화: ' ' -> '' (빈 문자열)로 변경
                'prompt': ''
            }
            # ▲▲▲ [버그 수정 4 완료] ▲▲▲
            
            response = requests.post(WHISPER_SERVER_URL, files=files, data=data)

            if response.status_code == 200:
                result_json = response.json()
                transcribed_text = result_json.get('text', '')
                
                cleaned_text = re.sub(r'\[.*?\]|\(.*?\)', '', transcribed_text).strip()
                print(f"Transcription Result: {cleaned_text}")
                return cleaned_text
            else:
                print(f"Whisper server returned error: {response.status_code}")
                print(f"Response: {response.text}")
                return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Whisper server: {e}")
        print("      (Is 'whisper-server.exe' running on port 8081?)")
        return None
    except Exception as e:
        print(f"An error occurred during STT request: {e}")
        return None
    finally:
        if tempfile.gettempdir() in normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)
# ▲▲▲ run_stt 함수 ▲▲▲


def process_and_send_keywords(audio_buffer):
    if not audio_buffer: return
    temp_wav_path = create_wav_file(bytes(audio_buffer), RATE, is_temp=True)
    text_result = run_stt(temp_wav_path)
    if text_result: # text_result가 비어있지 않은 경우에만 키워드 검사
        keywords_to_send = []
        with keywords_lock: current_target_keywords = TARGET_KEYWORDS[:]
        
        for keyword in current_target_keywords:
            if fuzz.partial_ratio(keyword, text_result) >= SIMILARITY_THRESHOLD:
                keywords_to_send.append(keyword)
        
        for alarm_keyword in ALARM_SOUND_KEYWORDS:
            if alarm_keyword in text_result.lower():
                if "siren" in alarm_keyword: keywords_to_send.append("siren")
                elif "horn" in alarm_keyword: keywords_to_send.append("horn")
                elif "boom" in alarm_keyword: keywords_to_send.append("boom")

        # (문제 2 해결됨)
        if keywords_to_send:
            response_message = ",".join(sorted(list(set(keywords_to_send))))
            print(f"\n[!!!] >>> Keywords DETECTED: {response_message} <<<\n")
            with socket_lock:
                if command_client_socket: 
                    try:
                        command_client_socket.sendall((response_message + '\n').encode('utf-8'))
                        print(f"[*] Successfully sent to app (via Command Port): {response_message}")
                    except socket.error as e:
                        print(f"[!] Failed to send keyword data: {e}.")
                else:
                    print(f"[!] Cannot send keyword data: Command client not connected.")

# ▼▼▼ handle_audio_client (단순 VAD 사용) ▼▼▼
def handle_audio_client(client_socket):
    global audio_client_socket, NOISE_THRESHOLD 
    with socket_lock: audio_client_socket = client_socket
    
    print("[*] Audio client connected. Starting VAD audio stream processing...")
    print(">>> Press 'g' to start a 5-second manual recording <<<")
    
    hp_filter = HighPassFilter()
    audio_buffer = bytearray()
    is_speaking = False
    silence_counter = 0
    manual_recording_buffer = bytearray()
    manual_recording_start_time = 0

    # ▼▼▼ [버그 수정 5] '음성'으로 감지된 청크의 수를 세는 카운터 추가 ▼▼▼
    speech_chunk_counter = 0
    
    try:
        while True:
            data = client_socket.recv(CHUNK * SAMPWIDTH)
            if not data: break
            
            filtered_data = hp_filter.process(data)
            rms = audioop.rms(filtered_data, SAMPWIDTH)
            print(f"RMS: {rms:<5}\r", end="")
            sys.stdout.flush()

            if manual_recording_triggered.is_set():
                if manual_recording_start_time == 0:
                    print("\n*** Starting 5-second manual recording... ***")
                    manual_recording_start_time = time.time()
                manual_recording_buffer.extend(filtered_data)
                
                if time.time() - manual_recording_start_time >= 5:
                    print("\n*** Finished manual recording. Saving file... ***")
                    create_wav_file(manual_recording_buffer, RATE, is_temp=False, filename_prefix="manual_rec_")
                    manual_recording_buffer.clear()
                    manual_recording_start_time = 0
                    manual_recording_triggered.clear()
                    print(">>> Press 'g' to start a new 5-second recording <<<")

            if rms > NOISE_THRESHOLD:
                if not is_speaking: print("\nSpeaking detected...")
                is_speaking = True
                silence_counter = 0
                audio_buffer.extend(filtered_data)
                
                # ▼▼▼ [버그 수정 5] '음성' 청크 카운트 증가 ▼▼▼
                speech_chunk_counter += 1
                
            elif is_speaking:
                silence_counter += 1
                audio_buffer.extend(filtered_data)
                
                if silence_counter > REQUIRED_SILENCE_CHUNKS:
                    # ▼▼▼ [버그 수정 5] STT 요청 전, 음성 청크의 수가 최소 기준(0.5초)을 넘는지 확인 ▼▼▼
                    if speech_chunk_counter >= REQUIRED_SPEECH_CHUNKS:
                        print(f"\nSpeech long enough ({speech_chunk_counter} chunks), processing speech...")
                        process_and_send_keywords(audio_buffer)
                    else:
                        print(f"\nSpeech too short ({speech_chunk_counter} chunks < {REQUIRED_SPEECH_CHUNKS}), ignoring as noise burst...")
                    # ▲▲▲ [버그 수정 5 완료] ▲▲▲

                    # 버퍼 및 상태 변수 초기화
                    audio_buffer.clear()
                    is_speaking = False
                    silence_counter = 0
                    speech_chunk_counter = 0 # 카운터 초기화

    except (ConnectionResetError, socket.error):
        print("\n[*] Audio client connection lost.")
    finally:
        with socket_lock: audio_client_socket = None
        print("\n[*] Audio client connection closed.")
# ▲▲▲ handle_audio_client ▲▲▲


def handle_command_client(client_socket):
    # (문제 2 해결됨)
    global TARGET_KEYWORDS, NOISE_THRESHOLD, command_client_socket
    print("[*] Command client connected.")
    with socket_lock:
        command_client_socket = client_socket # 전역 변수에 소켓 저장
        
    try:
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

                    # ▼▼▼ 수정된 코드 (민감도 범위 변경) ▼▼▼
                    # 1 (둔감) -> 2000 RMS, 10 (민감) -> 200 RMS
                    MIN_NOISE = 2000 # 둔감 (기존 500)
                    MAX_NOISE = 200  # 민감 (기존 50)
                    # ▲▲▲ 수정된 코드 ▲▲▲
                    
                    # 1(2000) ~ 10(200) 사이의 값을 매핑
                    mapped_threshold = MIN_NOISE + ((app_value - 1) / 9.0) * (MAX_NOISE - MIN_NOISE)
                    
                    NOISE_THRESHOLD = int(mapped_threshold) 
                    print(f"\n[!!!] NOISE SENSITIVITY updated by app: {app_value} (Threshold: {NOISE_THRESHOLD})\n")

                except Exception as e:
                    print(f"[!] Failed to parse sensitivity command: {e}")

    except Exception:
        print("[*] Command client connection lost.")
    finally:
        # (문제 2 해결됨)
        print("[*] Command client connection closed.")
        with socket_lock:
            command_client_socket = None # 소켓 연결 종료 시 전역 변수 초기화


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

def keyboard_listener():
    print("(Keyboard listener started)")
    keyboard.add_hotkey('g', lambda: manual_recording_triggered.set())
    while True:
        time.sleep(1)

def main():
    # ▼▼▼ whisper-server.exe 실행 확인 로직 추가 ▼▼▼
    try:
        response = requests.get(WHISPER_SERVER_URL.replace("/inference", "/health"))
        if response.status_code != 200 or response.json().get("status") != "ok":
             raise Exception("Server not healthy")
        print(f"[*] Connected to whisper.cpp server at {WHISPER_SERVER_URL}")
    except Exception as e:
        print(f"[!!!] FATAL ERROR: Cannot connect to whisper.cpp server at {WHISPER_SERVER_URL}")
        print("      Please ensure 'whisper-server.exe' is running (with -l ko) on port 8081.")
        sys.exit(1)
    # ▲▲▲ 수정 끝 ▲▲▲

    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)
    k_thread = threading.Thread(target=keyboard_listener)
    k_thread.daemon = True
    k_thread.start()
    audio_thread = threading.Thread(target=start_server, args=(AUDIO_PORT, handle_audio_client))
    command_thread = threading.Thread(target=start_server, args=(COMMAND_PORT, handle_command_client))
    audio_thread.daemon = True
    command_thread.daemon = True
    audio_thread.start()
    command_thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[*] Server is shutting down.")

if __name__ == "__main__":
    main()