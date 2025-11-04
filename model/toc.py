import torch
import torchvision # ResNet 클래스를 인식하려면 import 해주는 게 좋습니다.

# 확인하려는 .pt 파일 경로
file_path = r"C:\server\ICT_neckband-master\model\resnet50_best.pt"

print(f"'{file_path}' 파일 로드 중...")

try:
    # map_location='cpu'는 GPU가 없어도 안전하게 열기 위한 옵션입니다.
    # weights_only=False 를 추가하여 모델 클래스도 함께 로드하도록 허용합니다.
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

    print("--- 파일 로드 성공 ---")
    print(f"파일의 전체 데이터 타입: {type(checkpoint)}")

    # 1. 체크포인트가 딕셔너리(dict) 형태일 경우 (가장 일반적)
    if isinstance(checkpoint, dict):
        print("\n[파일 내용물 (딕셔너리 키 목록)]")
        print("---------------------------------")
        # 딕셔너리의 '키(key)'들만 먼저 출력해 봅니다.
        print(checkpoint.keys())
        print("---------------------------------")
        
        print("\n'할당값 번호' (클래스 맵)는 'class_to_idx', 'classes', 'label_map' 등의 키에 저장되어 있을 수 있습니다.")
        print("위 목록에 비슷한 키가 있는지 확인해 보세요.")

        # 예: 만약 'class_to_idx' 라는 키가 있다면, 그 내용을 직접 확인
        key_to_find = 'class_to_idx' # 또는 'classes' 등 찾고 싶은 키 이름
        if key_to_find in checkpoint:
            print(f"\n--- 찾은 키 '{key_to_find}'의 내용 ---")
            print(checkpoint[key_to_find])
        
    # 2. 딕셔너리가 아닐 경우 (예: ResNet 모델 객체 자체가 저장된 경우)
    else:
        print("\n이 파일은 딕셔너리가 아니라, 아마도 모델 객체 자체인 것 같습니다.")
        # 모델 객체라면, 'classes' 같은 속성을 가지고 있을 수 있습니다.
        if hasattr(checkpoint, 'classes'):
            print("\n--- 모델 객체에서 'classes' 속성을 찾았습니다! ---")
            print(checkpoint.classes)
        elif hasattr(checkpoint, 'class_to_idx'):
            print("\n--- 모델 객체에서 'class_to_idx' 속성을 찾았습니다! ---")
            print(checkpoint.class_to_idx)
        else:
            print("모델 객체에서 'classes' 또는 'class_to_idx' 속성을 찾지 못했습니다.")


except Exception as e:
    print(f"\n--- 에러 발생 ---")
    print(f"파일을 로드하는 데 실패했습니다: {e}")
    print("파일 경로가 정확한지, PyTorch가 제대로 설치되었는지 확인해 보세요.")