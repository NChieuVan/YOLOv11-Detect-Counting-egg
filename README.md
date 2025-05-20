# YOLOv11-Detect-Counting-egg
## Object Counting trên Video với Ultralytics và OpenCV
- Đếm số lượng trứng đã qua đường kẽ cho trước
## Mô tả
Project này sử dụng thư viện [ultralytics](https://github.com/ultralytics/ultralytics) để đếm số lượng đối tượng trong video dựa trên một vùng quan tâm (region of interest) định nghĩa bằng đa giác.  
Kết quả sẽ được ghi lại thành video mới với các đối tượng được đếm và hiển thị vùng đếm.

## Kết quả demo
![demo](https://github.com/user-attachments/assets/5f1ce539-4e6e-4472-b465-c6a0236024d0)

## Trainning model yolo
- Dataset : https://www.kaggle.com/datasets/nguynchiuvn/detect-egg
- ![result-train](https://github.com/user-attachments/assets/81c62043-c492-4ca8-95d9-4c4ca947e143)
- ![results](https://github.com/user-attachments/assets/9fa0b37a-050b-4269-9c37-90db2acbc219)
- ![confusion_matrix](https://github.com/user-attachments/assets/ddb3ac85-e093-49c4-8585-47ec8331188b)

## Yêu cầu
- Python 3.7 trở lên
- OpenCV (`cv2`)
- Ultralytics (cài đặt qua `pip install ultralytics`)

## Cách sử dụng trên Google Colab hoặc Local

1. Upload video nguồn vào thư mục `/content/` (hoặc mount Google Drive)
2. Upload file model YOLO `.pt` đã huấn luyện (ví dụ: `best.pt`) vào `/content/`
3. Chạy đoạn code xử lý video, ví dụ:

```python
import cv2
from ultralytics import solutions
from IPython.display import Video

cap = cv2.VideoCapture("/content/2025-05-21 00-34-35.mp4")
assert cap.isOpened(), "Error reading video file"

region_points = [[523, 4], [962, 604], [1117, 585], [647, 1]]

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

video_writer = cv2.VideoWriter("object_counting_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

counter = solutions.ObjectCounter(
    show=False,
    region=region_points,
    model="/content/best.pt",
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break
    results = counter(im0)
    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

Video("object_counting_output.mp4", embed=True)
