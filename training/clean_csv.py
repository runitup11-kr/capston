import os
import pandas as pd

# 🌟🌟 1. 경로 설정 (train_pilotnet.py에서 사용한 경로와 일치시켜야 합니다!) 🌟🌟
DATASET_ROOT = r"C:\Users\USER\Desktop\dataset"
OLD_CSV_FILENAME = "data_labels.csv"
NEW_CSV_FILENAME = r"C:\Users\USER\Desktop\dataset\data_labels_cleaned.csv"

# ----------------------------------------------------

# 2. CSV 파일의 절대 경로 생성
csv_filepath = os.path.join(DATASET_ROOT, OLD_CSV_FILENAME)

try:
    # 3. 기존 CSV 파일 읽기
    df_labels = pd.read_csv(csv_filepath)
    print(f"✅ 기존 CSV 파일 ({OLD_CSV_FILENAME}) 로드 완료. 총 {len(df_labels)}개 항목.")

    # 4. 실제 디스크에 있는 파일 목록을 불러오기
    # os.listdir(DATASET_ROOT)는 폴더 내의 모든 파일 이름을 리스트로 반환합니다.
    actual_files = set(os.listdir(DATASET_ROOT))
    print(f"✅ 데이터셋 폴더 ({DATASET_ROOT})에서 실제 파일 {len(actual_files)}개 확인.")

    # 5. CSV 목록의 파일이 실제 존재하는지 확인
    # 'filename' 컬럼에 이미지 파일 이름이 있다고 가정합니다. (예: '20251202_...png')
    
    # 실제 파일 목록에 포함된 행만 필터링합니다.
    # CSV 파일의 'filename' 컬럼 이름이 다를 경우, 이 부분을 수정해야 합니다. 
    # 예를 들어, CSV에 'image_path'라고 되어 있다면, df_labels['image_path']로 수정
    
    df_cleaned = df_labels[df_labels['filename'].isin(actual_files)]

    # 6. 정리된 파일 수 확인
    count_deleted = len(df_labels) - len(df_cleaned)
    print(f"❌ 불일치 또는 삭제된 항목 {count_deleted}개 제거.")
    print(f"👍 정리 후 유효한 항목 {len(df_cleaned)}개 남음.")

    # 7. 새로운 CSV 파일로 저장
    new_csv_filepath = os.path.join(DATASET_ROOT, NEW_CSV_FILENAME)
    df_cleaned.to_csv(new_csv_filepath, index=False)
    print(f"\n🌟🌟 새 CSV 파일 '{NEW_CSV_FILENAME}' 저장이 완료되었습니다. 🌟🌟")
    print(f"새 파일 경로: {new_csv_filepath}")

except FileNotFoundError:
    print(f"🚨 오류: CSV 파일을 찾을 수 없습니다. 경로를 확인하세요: {csv_filepath}")
except KeyError:
    print("🚨 오류: CSV 파일에 'filename' 컬럼이 없습니다. CSV 파일의 이미지 파일명 컬럼 이름을 확인하고 코드('filename')를 수정하세요.")
except Exception as e:
    print(f"🚨 예상치 못한 오류 발생: {e}")