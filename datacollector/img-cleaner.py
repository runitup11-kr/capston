import os
import pandas as pd

# 데이터셋 루트 경로
dataset_root = r"C:\Users\USER\Desktop\dataset"

# 기존 CSV 파일 경로 (오류 수정)
csv_file = os.path.join(dataset_root, "data_labels.csv") # 🌟 os.path.join 사용!

# 새로운 CSV 파일 경로 (오류 수정)
new_csv_file = os.path.join(dataset_root, "data_labe_update.csv") # 🌟 os.path.join 사용!

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 이미지 파일 존재 여부 체크 함수
def file_exists(filename):
    file_path = os.path.join(dataset_root, filename)
    return os.path.exists(file_path)

# 존재하는 파일만 필터링하여 새로운 CSV 파일 생성
valid_rows = []

for _, row in df.iterrows():
    img_path = row['image_path']  # 'image_path' 컬럼으로 수정
    if file_exists(img_path):
        valid_rows.append(row)
    else:
        print(f"[WARN] Missing file: {img_path}")

# 새로운 CSV 파일 생성
valid_df = pd.DataFrame(valid_rows)
valid_df.to_csv(new_csv_file, index=False)

# 출력: 전체 파일 수 및 각 각도별 이미지 수와 비율
total_images = len(valid_df)
print(f"[INFO] Total valid images: {total_images}")

# 각 각도별 이미지 수와 비율 계산
angle_counts = valid_df['servo_angle'].value_counts()
angle_percentages = (angle_counts / total_images) * 100

print("\n[INFO] Image counts and percentages by angle:")
for angle, count in angle_counts.items():
    percentage = angle_percentages[angle]
    print(f"Angle: {angle}, Count: {count}, Percentage: {percentage:.2f}%")
