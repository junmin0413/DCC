import os
import shutil
import random

# 이미지 확장자 목록 (필요 시 추가)
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".PNG", ".JPEG"]

def _find_image(images_dir, stem):
    """stem과 일치하는 이미지 파일을 images_dir에서 탐색"""
    for ext in IMAGE_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def split_dataset_with_labels(
    images_dir,         # 이미지 폴더 (예: TS_xz_train/images 또는 TS_xz_train)
    labels_dir,         # 라벨 폴더 (예: TS_xz_train/labels)
    output_dir,         # 출력 루트 (여기에 images/labels 하위 train/val/test 생성)
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    label_ext=".txt",
    seed=42
):
    """
    이미지/라벨을 동일 파일명(stem) 기준으로 페어링하여 같은 subset으로 분할해줌
    출력 구조:
      output_dir/
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
    """
    # 비율 체크
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError("train/val/test 비율 합이 1이어야 함")

    random.seed(seed)

    # 출력 폴더 생성
    for subset in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", subset), exist_ok=True)

    # 1) 라벨 기준으로 페어링
    label_files = [f for f in os.listdir(labels_dir)
                   if os.path.isfile(os.path.join(labels_dir, f)) and f.endswith(label_ext)]
    pairs = []
    missing_image = 0

    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            missing_image += 1
            continue
        lbl_path = os.path.join(labels_dir, lf)
        pairs.append((img_path, lbl_path))

    # 2) 셔플 & 스플릿
    random.shuffle(pairs)
    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    # 3) 복사 함수
    def copy_pair_list(pair_list, subset):
        img_out_dir = os.path.join(output_dir, "images", subset)
        lbl_out_dir = os.path.join(output_dir, "labels", subset)
        for img_path, lbl_path in pair_list:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            img_ext = os.path.splitext(img_path)[1]
            shutil.copy(img_path, os.path.join(img_out_dir, stem + img_ext))
            shutil.copy(lbl_path, os.path.join(lbl_out_dir, stem + label_ext))

    # 4) 복사 실행
    copy_pair_list(train_pairs, "train")
    copy_pair_list(val_pairs, "val")
    copy_pair_list(test_pairs, "test")

    # 리포트
    print("=== Split Report ===")
    print(f"총 페어링 수(라벨 기준): {total}")
    print(f" - train: {len(train_pairs)}")
    print(f" - val  : {len(val_pairs)}")
    print(f" - test : {len(test_pairs)}")
    print(f"라벨은 있는데 매칭 이미지 없음: {missing_image}")

# ✅ 사용 예시
# (예시) 폴더가 다음처럼 있을 때:
# TS_xz_train/images/*.jpg
# TS_xz_train/labels/*.txt
# ====== 여기만 정확히 맞추세요 ======
images_dir = r"/Users/jangjunmin/Desktop/DCC/project/GPR_dataset/Training/Source_data/TS_xz_train"
labels_dir = r"/Users/jangjunmin/Desktop/DCC/project/GPR_dataset/Training/labeled_data/TL_xz_train"
output_dir = r"/Users/jangjunmin/Desktop/DCC/project/xz_dataset"  # 맥 경로 예시

split_dataset_with_labels(
    images_dir=images_dir,
    labels_dir=labels_dir,
    output_dir=output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    label_ext=".json",     # ★ 라벨 확장자 변경
    seed=42
)
