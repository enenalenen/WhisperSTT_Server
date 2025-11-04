import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

def logmel(audio,
           num_samples,
           sample_rate=16000,
           n_fft=512,
           hop_length=160,
           win_length=512,
           n_mels=64,
           fmin=50,
           fmax=8000,
           top_db=80
           ):
    
    """
    파일 → [1, 1, n_mels, 101] Log-Mel 스펙트로그램 NumPy (float32)

    Args
    ----
    file_path  : 오디오 파일 경로
    time       : 고정할 파형 길이(샘플 수). 부족하면 0-padding, 초과하면 자름
    sample_rate: 목표 샘플링 주파수
    top_db     : 동적 범위(‘최대 - top_db’) 이하를 자르는 dB 클리핑
                 None이면 클리핑 생략
    """

    if isinstance(audio, np.ndarray):
        # ndarray 입력일 때 처리
        waveform = torch.from_numpy(audio.flatten())
        sr = sample_rate
    else:
        # raw파일일때 처리
        waveform, sr = torchaudio.load(audio)

    # (1-a) 1차원 → 2차원으로 변환 (채널 축 추가)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # (1-b) 스테레오 등 멀티채널이면 평균 → Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # (1-c) 필요 시 Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # 길이 보정(패딩/자르기)
    if waveform.shape[1] < num_samples:
        waveform = F.pad(waveform, (0, num_samples - waveform.shape[1]))
    else:
        waveform = waveform[:, :num_samples]

    # 최대값 정규화
    waveform = waveform / (waveform.abs().max() + 1e-9)

    # 복소 STFT [1, F, T']
    window = torch.hann_window(win_length, device=waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, pad_mode="reflect", return_complex=True)
    magnitude = torch.abs(stft) # 진폭 스펙트럼
    
    # 멜 스펙트로그램 [1, 64, T']
    melfilter = T.MelScale(n_mels=n_mels, sample_rate=sample_rate, f_min=fmin, f_max=fmax, n_stft=magnitude.shape[1], norm=None, mel_scale="htk").to(magnitude.device)
    mel = melfilter(magnitude)

    # 로그 스케일
    logmel = torch.log(torch.clamp(mel, min=1e-10))

    # 다이나믹 레인지 동적 범위 컷(top_db dB 이하는 절단)
    if top_db is not None:
        logmel = torch.maximum(logmel, logmel.max(dim=-1, keepdim=True).values - top_db)

    # 배치 차원 추가 [B, 1, 64, T']
    logmel = logmel.unsqueeze(0)

    # 시간축(T')을 101 프레임으로 고정
    if logmel.shape[-1] < 101:
        logmel = F.pad(logmel, (0, 101 - logmel.shape[-1]))

    elif logmel.shape[-1] > 101:
        logmel = logmel[:, :, :101]

    return logmel.numpy()

def scale(logmel_np, uint8=False):

    x = logmel_np

    # 전체 배열에서 최솟값을 0으로 이동 (shift)
    x -= x.min()

    # 최대값을 1로 맞춤 (scale)
    # +1e-9 는 분모가 0이 되는 예외를 막기 위한 작은 상수
    x /= x.max() + 1e-9

    # 0~1 → 0~255 선형 확장
    x *= 255.0

    # 소숫점 반올림 후 uint8 캐스팅
    if uint8:
        return np.round(x).astype(np.uint8)
    else:
        return x

# 과적합 방지용 시간/주파수 마스킹(학습 전용)
def spec_augmentation(logmel, time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2):
    """
    logmel: [channel, mel_bins, time]
    - time_drop_width: 시간 마스킹 폭
    - time_stripes_num: 시간 마스킹 반복 횟수
    - freq_drop_width: 주파수 마스킹 폭
    - freq_stripes_num: 주파수 마스킹 반복 횟수
    """
    freq_mask = T.FrequencyMasking(freq_mask_param=freq_drop_width)
    time_mask = T.TimeMasking(time_mask_param=time_drop_width)

    for _ in range(freq_stripes_num):
        logmel_aug = freq_mask(logmel)

    for _ in range(time_stripes_num):
        logmel_aug = time_mask(logmel)
    
    return logmel_aug