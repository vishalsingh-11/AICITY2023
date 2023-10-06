import cv2
from tqdm import tqdm
# Parse the data from file and convert it to dictionary format: {frame_number: [(bounding box, class_id)]}
def parse_data_from_file(filename):
    boxes = {}
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(',')
            frame = int(parts[0])
            class_id = int(parts[1])
            x1, y1, w, h = map(float, parts[2:6])
            if frame not in boxes:
                boxes[frame] = []
            boxes[frame].append(((x1, y1, x1+w, y1+h), class_id))
    return boxes
# Draw bounding boxes and class ids on frames
def draw_boxes_on_video(video_path, boxes, output_path):
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, 29.0, (int(cap.get(3)), int(cap.get(4))))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 0
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 0, 0)
    font_thickness = 2
    # Use tqdm for progress bar
    for _ in tqdm(range(total_frames), desc="Processing frames", ncols=100):
        ret, frame = cap.read()
        if not ret:
            continue
        # If there are boxes for the current frame, draw them
        if frame_number in boxes:
            for box, class_id in boxes[frame_number]:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, str(class_id), (int(box[0]), int(box[1]-10)), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        out.write(frame)
        frame_number += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# data_file = '../data/hungarian_cluster_S001/S001_c001.txt'  
# data_file = '../data/SCT/S001_c001.txt'
# data_file = '../data/SCT/S003_c014.txt'
# data_file = '../data/final_n=15_dist200_pk_filter_margin_2/S001_c001.txt'
# data_file = '/data/rbondili/aicity_tracking/AICITY2023/reproduce/S001_c001.txt'
data_file = '/data/rbondili/AICITY2023/AIC23_Track1_MTMC_Tracking/validation/S005/c025/label.txt'
boxes = parse_data_from_file(data_file)
# video_path = '../AIC23_Track1_MTMC_Tracking/test/S001/c001/video.mp4'  # Input video path
video_path = '/data/rbondili/AICITY2023/AIC23_Track1_MTMC_Tracking/validation/S005/c025/video.mp4'
# video_path = '../AIC23_Track1_MTMC_Tracking/test/S003/c014/video.mp4'
output_video_path = "viz_outputs/output_S005_c025_Syn_test.mp4"  # Output video path
draw_boxes_on_video(video_path, boxes, output_video_path)







