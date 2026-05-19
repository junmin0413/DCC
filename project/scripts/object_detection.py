import os
import json
import cv2
from collections import defaultdict

# 파일 찾기.
def find_files(folder_path, extensions):
    files_dict = {}
    # 폴더 경로가 존재하는지 확인(없으면 종료).
    if not os.path.exists(folder_path):
        print(f"경고: 폴더가 존재하지 않습니다: {folder_path}")
        return files_dict
    #folder_path 아래 하위 디렉토리를 재귀적으로 탐색.
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # 파일 확장자가 extensions 튜플에 지정된 확장자인지 확인.
            if filename.lower().endswith(extensions):
                # 확장자 제거, 순수한 이름만 추출.
                base_name = os.path.splitext(filename)[0]  # 확장자 없는 파일명
                files_dict[base_name] = os.path.join(root, filename)
    return files_dict

# json 파일 로드, 바운딩 박스 정보와 라벨 추출.
def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            # json 파일 로드.
            data = json.load(f)
            # json 파일 annotation 키가 있고, annotation 값이 dict 타입인지 확인.
            if "annotation" in data and isinstance(data["annotation"], dict):
                annotation = data["annotation"]
                # annotation에서 classes 키 값의 라벨 정보 추출
                label = annotation.get("classes")
                
                # bbox 좌표 정보 추출
                bbox_x = annotation.get("bbox_x")
                bbox_y = annotation.get("bbox_y")
                bbox_w = annotation.get("bbox_w")
                bbox_h = annotation.get("bbox_h")
                
                # 라벨과 좌표가 존재하는지 확인.
                if all([label, bbox_x, bbox_y, bbox_w, bbox_h]):
                    return {
                        "objects": [{
                            "label": label,
                            "bbox": {
                                "x": int(bbox_x),
                                "y": int(bbox_y),
                                "w": int(bbox_w),
                                "h": int(bbox_h)
                            }
                        }]
                    }
            print(f"경고: {json_path} 파일에서 유효한 'annotation' 또는 bbox 정보가 없습니다.") # 오류 처리.
            return None
    # 오류 처리.
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류 {json_path}: {e}")
        return None
    except ValueError as e:
        print(f"바운딩 박스 좌표 변환 오류 {json_path}: {e}")
        return None
    except Exception as e:
        print(f"파일 읽기 또는 처리 오류 {json_path}: {e}")
        return None

# 바운딩 박스와 라벨 그림. labels_to_draw의 해당 라벨만 선택적으로 그림.
def draw_annotations(image, objects, color_map, labels_to_draw=None):
    if image is None:
        print("경고: 유효하지 않은 이미지 객체입니다.")
        return

    for obj in objects: # 'objects'는 이제 load_json_data에서 생성한 리스트
        label = obj.get('label')
        bbox = obj.get('bbox')

        # main 함수에서 특정 객체(예: 'bounding box') 종류만 시각화할때 사용.
        if labels_to_draw and label not in labels_to_draw:
            continue
            
        # 라벨이 color_map에 있고 bbox 정보가 유효한지 확인.
        if label in color_map and bbox:
            try:
                # 좌표 정보 가져오기.
                x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['w']), int(bbox['h'])
                color = color_map[label]
                
                # 바운딩 박스 그리기.
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # 라벨 배경 및 텍스트를 더 잘 보이도록 그리기
                label_text = str(label)
                # label text 크기 측정해 배경 사각형을 그릴 크기 결정.
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Y 좌표가 음수가 되지 않도록 보호함.
                bg_start_y = max(0, y - label_size[1] - 10)
                bg_end_y = max(0, y)
                text_y = max(10, y - 5) # 텍스트가 잘 보이도록 최소 y값 설정
                
                cv2.rectangle(image, (x, bg_start_y), (x + label_size[0], bg_end_y), color, -1)
                cv2.putText(image, label_text, (x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # 오류 처리.
            except (ValueError, KeyError) as e:
                print(f"바운딩 박스 그리기 오류 ({label} 객체): {e}")
            except Exception as e:
                print(f"예상치 못한 그리기 오류 ({label} 객체): {e}")
        elif label is not None and label not in color_map:
            print(f"디버그: '{label}' 라벨은 color_map에 정의되지 않았습니다.")
        elif bbox is None:
            print(f"디버그: '{label}' 객체에 바운딩 박스 정보가 없습니다.")


# 시작. 함수 호출하여 전체적인 파이프라인 실행.
def main():
    image_folder = "/Users/jangjunmin/Desktop/DCC/project/GPR_dataset/Training/Source_data/TS_xz_train"
    labeled_folder = "/Users/jangjunmin/Desktop/DCC/project/GPR_dataset/Training/labeled_data/TL_xz_train"
    
    # 테두리 색상 맥핑
    color_map = {
        'bounding box': (0, 0, 255),             # 빨간색
        'bounding pipe': (0, 255, 0),            # 초록색  
        'bounding patch': (255, 0, 0),           # 파란색
        'bounding manhole': (0, 255, 255),       # 노란색
        'bounding cavity': (255, 0, 255)         # 마젠타
    }
    
    print("--- 파일 검색 시작 ---")
    
    # 파일 찾기.
    image_files = find_files(image_folder, ('.jpg', '.jpeg', '.png'))
    json_files = find_files(labeled_folder, ('.json',))
    
    print(f"이미지 파일 {len(image_files)}개, JSON 파일 {len(json_files)}개 발견")
    
    # 매칭된 파일 찾기.
    matched_files = []
    for base_name in image_files:
        if base_name in json_files:
            matched_files.append((base_name, image_files[base_name], json_files[base_name]))
    
    print(f"{len(matched_files)}개의 매칭된 파일 쌍 발견")
    
    if not matched_files:
        print("매칭된 파일이 없습니다. 프로그램을 종료합니다.")
        return
    
    # 각 객체 타입별로 처음 나타나는 파일에서 시각화
    visualized_objects = set()
    object_file_map = defaultdict(list)  # 각 객체가 어떤 파일에 있는지 추적
    
    # 모든 파일을 스캔하여 객체 분포 파악
    print("\n--- 객체 분포 분석 중 ---")
    for base_name, image_path, json_path in matched_files:
        data = load_json_data(json_path) # 수정된 load_json_data 호출
        if data and 'objects' in data: # 'objects' 키가 있는지 확인
            for obj in data['objects']: # load_json_data에서 생성한 'objects' 리스트 사용
                label = obj.get('label')
                if label in color_map:
                    object_file_map[label].append((base_name, image_path, json_path))
                elif label is not None:
                    print(f"경고: JSON 라벨 '{label}'이 color_map에 정의되지 않았습니다. color_map을 확인하세요!")

    # 각 객체 타입별 파일 수 출력
    for label, files in object_file_map.items():
        print(f"'{label}': {len(files)}개 파일에서 발견")
    
    print(f"\n--- 시각화 시작 (총 {len(color_map)}가지 객체 타입 중 각 1개씩) ---")
    
    # 각 객체 타입별로 첫 번째 파일에서 시각화
    for label_to_check in color_map: # color_map의 라벨 순서대로 시도
        # 이미 시각화되었거나, 데이터셋에 해당 라벨이 없으면 건너뜁니다.
        if label_to_check in visualized_objects or label_to_check not in object_file_map:
            continue
        
        # 해당 라벨을 포함하는 첫 번째 파일을 가져옵니다.
        base_name, image_path, json_path = object_file_map[label_to_check][0]
        
        print(f"\n '{label_to_check}' 객체 시각화: {os.path.basename(image_path)}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"경고: 이미지를 읽을 수 없습니다: {image_path}. 다음 객체로 넘어갑니다.")
            continue
        
        # JSON 데이터 로드 (수정된 load_json_data 함수 사용)
        data = load_json_data(json_path)
        if not data:
            print(f"경고: JSON 데이터를 로드할 수 없습니다: {json_path}. 다음 객체로 넘어갑니다.")
            continue
        
        # `draw_annotations` 함수를 사용하여 해당 객체 타입만 그립니다.
        draw_annotations(image, data.get('objects', []), color_map, {label_to_check})
        
        # 이미지 표시
        window_name = f'GPR Visualization - {label_to_check}'
        cv2.imshow(window_name, image)
        
        print(f"'{label_to_check}' 객체가 표시된 이미지를 확인하세요. 창을 닫거나 아무 키나 누르면 다음으로 진행합니다.")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name) # 개별 창 닫기
        
        visualized_objects.add(label_to_check)
        
    print(f"\n--- 시각화 완료: 총 {len(visualized_objects)}가지 객체 타입을 확인했습니다. ---")
    cv2.destroyAllWindows() # 모든 OpenCV 창 닫기

if __name__ == "__main__":
    main()