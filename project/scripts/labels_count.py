import os
import json
from collections import Counter

# labeled_data 폴더 경로
base_folder = "/Users/jangjunmin/Desktop/DCC/project/GPR_dataset/Training/labeled_data"

all_classes = []

# 하위 폴더까지 모두 탐색
for root, dirs, files in os.walk(base_folder):
    for file_name in files:
        # JSON 파일만 처리
        if file_name.lower().endswith(".json"):
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cls = data.get("annotation", {}).get("classes")
                    if cls:
                        all_classes.append(cls)
            except Exception as e:
                print(f"{file_path} 처리 중 오류 발생: {e}")

# 결과 출력
unique_classes = set(all_classes)
print("객체 종류:", unique_classes)
class_counts = Counter(all_classes)
print("클래스별 개수:", class_counts)
