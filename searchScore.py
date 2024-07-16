from concurrent.futures import ThreadPoolExecutor
import glob
import pickle
import torch
import cv2
import numpy as np
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from collections import defaultdict
import os

# 加载模型
model_note = torch.jit.load('./models/score_note_320.torchscript.pt')
model_region = torch.jit.load('./models/region.torchscript.pt')
model_note.eval()
model_region.eval()

# 图像预处理函数
def preprocess_image(image, input_size):
    if image is None or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Invalid image dimensions.")
    img_resized, ratio, pad = letterbox(image, input_size, auto=False)
    img_resized = img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img_resized = np.ascontiguousarray(img_resized)
    img_tensor = torch.from_numpy(img_resized).float().div(255.0).unsqueeze(0)
    return img_tensor, ratio, pad

# 坐标变换函数
def scale_coords(pred, dw, dh, width, height, width_orig, height_orig):
    for item in pred:
        item[0] = (item[0] - dw) / (width - 2 * dw)
        item[2] = (item[2] - dw) / (width - 2 * dw)
        item[1] = (item[1] - dh) / (height - 2 * dh)
        item[3] = (item[3] - dh) / (height - 2 * dh)

        item[0] = int(item[0] * width_orig)
        item[1] = int(item[1] * height_orig)
        item[2] = int(item[2] * width_orig)
        item[3] = int(item[3] * height_orig)
    return pred

# 检测小节
def detect_regions(image, model, input_size):
    img_tensor, ratio, pad = preprocess_image(image, input_size)
    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
    if len(pred) > 0:
        pred = pred[0].cpu()
        pred = scale_coords(pred, pad[0], pad[1], img_tensor.shape[3], img_tensor.shape[2], image.shape[1], image.shape[0])
        # 排序：先按y坐标排序，再按x坐标排序
        pred = sorted(pred, key=lambda x: (x[1], x[0]))
        
        # 编号
        for i, region in enumerate(pred):
            region = list(region)
            region.append(i)
            pred[i] = region
    return pred


# 检测音符
def detect_notes(image, model, input_size):
    img_tensor, ratio, pad = preprocess_image(image, input_size)
    with torch.no_grad():
        pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
    if len(pred) > 0:
        pred = pred[0].cpu()
        pred = scale_coords(pred, pad[0], pad[1], img_tensor.shape[3], img_tensor.shape[2], image.shape[1], image.shape[0])

        # 绘制检测结果
        for note in pred:
            x1, y1, x2, y2 = int(note[0]), int(note[1]), int(note[2]), int(note[3])
            conf = note[4]
            cls = int(note[5])

            # 绘制框和标签
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"{cls}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return pred

# 加载图像
def detect_and_draw_notes(img_path, model_note, model_region, note_input_size=320, region_input_size=640):
    # img_path = './region.jpg'
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read image from {img_path}. Please check the file path and ensure the file exists.")
        exit()

    # 检测小节
    regions = detect_regions(img, model_region, region_input_size)
    # print("Detected regions:", regions)

    # 切分图像并检测每个小节中的音符
    all_notes = []
    total_cord=0
    for i, region in enumerate(regions):
        x1, y1, x2, y2, conf, cls,idx = region
        x1=int(x1)
        x2=int(x2)
        y1=int(y1)-30
        y2=int(y2)+30

        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        # cv2.imshow('Image', cropped_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            print(f"Warning: Skipping region {i} due to invalid dimensions.")
            continue
        notes = detect_notes(cropped_img, model_note, note_input_size)
        # cv2.imshow('Image', cropped_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # chords=defaultdict(list)
        
        left_hand, right_hand = separate_hands(notes,region)
        chords_left=group_chords(left_hand,3)
        chords_right=group_chords(right_hand,3)
        chords_binary_l=chords_to_binary(chords_left,i,25,False)
        chords_binary_r=chords_to_binary(chords_right,i,25)
        total_cord=total_cord+len(chords_binary_l)+len(chords_binary_r)
        # chords[i].extend(chords_binary_l)
        # chords[i].extend(chords_binary_r)#区域号，和弦索引，和弦二进制表示
        # all_notes.extend([(i, chord_idx, chord) for chord_idx, chord in enumerate(chords_binary_l)])
        # all_notes.extend([(i, chord_idx, chord) for chord_idx, chord in enumerate(chords_binary_r)])
        all_notes.extend(chords_binary_l)
        all_notes.extend(chords_binary_r)
        # print(all_notes)
        # for note in notes:
        #     # 将音符坐标变换回原始图像坐标
        #     note=note.tolist()
        #     note.append(region)
        #     note[0] += x1
        #     note[1] += y1
        #     note[2] += x1
        #     note[3] += y1
        #     # print(f"X:{note[0]}{note[2]},Y:{note[1]}{note[3]},region:{region}")
        #     all_notes.append(note)
    # cv2.imshow('Image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return all_notes,total_cord

# 分离左右手音符
def separate_hands(notes,region):
    left_hand = []
    right_hand = []
    for note in notes:
        region_top = region[1]-30
        region_bottom = region[3]+30
        region_middle = (region_top + region_bottom) / 2

        y_center = (note[1] + note[3]) / 2
        note_label = note[5]
        if y_center < region_middle and note_label <= 5:
            left_hand.append(note)
        elif y_center >= region_middle and note_label >= 23:
            right_hand.append(note)
        else:

            if y_center < region_middle:
                right_hand.append(note)
            elif y_center > region_middle:
                left_hand.append(note)     

    return left_hand, right_hand


# 将音符分组为和弦
def group_chords(notes, x_threshold=3):
    chords = defaultdict(list)
    sorted_notes = []
    sorted_notes = sorted(notes, key=lambda note: ((note[0] + note[2]) / 2,(note[1] + note[3]) / 2 ))
    for note in sorted_notes:
        x_center = (note[0] + note[2]) / 2  # 计算音符的水平中心位置
        assigned = False
        for key in chords.keys():
            if abs(key - x_center) < x_threshold:
                chords[key].append(note)  
                assigned = True
                break
        if not assigned:
            chords[x_center].append(note)  
     # 对和弦进行重新编号，将key:x_center映射为从0开始的自然数
    sorted_chords = sorted(chords.items())
    renumbered_chords = {i: chord for i, (_, chord) in enumerate(sorted_chords)}

    return renumbered_chords


# 将和弦转换为二进制表示
def chords_to_binary(chords,region_id, num_lines=25,right=True):
    binary_chords = []
    for index, chord in chords.items():
        binary_rep = [0] * num_lines
        for note in chord:
            note_label = int(note[5])  # 确保 note_label 是整数
            if 0 <= note_label < num_lines:
                binary_rep[note_label] = 1  # 将标号转换为二进制表示
        if right:
            chord_binary = int("".join(map(str, binary_rep))+'1', 2)  # 将二进制数组转换为整数
        else:
            chord_binary = int("".join(map(str, binary_rep))+'0', 2)
        binary_chords.append((region_id,index, chord_binary))  # 区域中和弦索引，二进制表示
    return binary_chords


# 用于存储图像特征的数据库
class ImageDatabase:
    def __init__(self):
        self.db = defaultdict(list)  # 反向索引数据库
        self.track_info = {}  # 存储图像ID到曲目名称的映射
        self.inverted_index = defaultdict(list)  # 反向索引

    def add_image(self, image_id, binary_chords, track_name):
        self.track_info[image_id] = track_name  # 存储图像ID和曲目名称
        for region_idx, chord_idx, chord in binary_chords:
            chord_tuple = chord
            self.db[chord_tuple].append((image_id, region_idx, chord_idx))
            self.inverted_index[chord_tuple].append((region_idx, chord_idx))

    def search_image(self, query_chords, tolerance=1):
        matches = defaultdict(lambda: defaultdict(int))  # matches[image_id][region_idx] = count
        for q_region_idx, q_chord_idx, q_chord in query_chords:
            q_chord_tuple = (q_chord)
            if q_chord_tuple in self.db:
                for image_id, region_idx, chord_idx in self.db[q_chord_tuple]:
                    # print(f"{image_id}\n")
                    if abs(q_region_idx - region_idx) <= tolerance and abs(q_chord_idx - chord_idx) ==0:
                        matches[image_id][region_idx] += 1
        return matches

    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump((self.db, self.track_info, self.inverted_index), f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            self.db, self.track_info, self.inverted_index = pickle.load(f)


#多线程处理
def process_image(img_path):
    # 检测和绘制音符，并返回所有和弦的二进制表示
    print(f"process{img_path}\n")
    all_chords, totalcord = detect_and_draw_notes(img_path, model_note, model_region)
    return img_path, all_chords, totalcord
#引入多线程处理
def main():
    search_folder = './searchjpg'
    database_path = './music_db.pkl'
    result_file = './result.txt'

    image_db = ImageDatabase()
    if os.path.exists(database_path):
        image_db.load(database_path)
    else:
        print(f"Error: Database file {database_path} does not exist.")
        return

    img_paths = glob.glob(os.path.join(search_folder, '*.jpg'))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, img_paths))

    with open(result_file, 'w') as f:
        for img_path, all_chords, totalcord in results:
            matches = image_db.search_image(all_chords, tolerance=0)
            ratios = {}
            for img_id, regions_id in matches.items():
                match_cord = sum(regions_id.values())
                ratios[img_id] = match_cord / totalcord

            top_matches = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:3]
            top_track_names = [image_db.track_info[image_id] for image_id, _ in top_matches]

            if top_track_names:
                f.write(f"Image: {os.path.basename(img_path)}\n")
                for id, ratio in top_matches:
                    f.write(f"ID: {image_db.track_info[id]}\t\tRatio: {ratio:.2f}\n")
                f.write("\n")
            else:
                f.write(f"Image: {os.path.basename(img_path)}\nNot found.\n\n")
            
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    main()

# def main():
#     search_folder = './searchjpg'  # 搜索文件夹
#     database_path = './music_db.pkl'
#     result_file = './result.txt'

#     # 加载数据库
#     image_db = ImageDatabase()
#     if os.path.exists(database_path):
#         image_db.load(database_path)
#     else:
#         print(f"Error: Database file {database_path} does not exist.")
#         return

#     with open(result_file, 'w') as f:
#         for img_path in glob.glob(os.path.join(search_folder, '*.jpg')):
#             print(f"Processing {img_path}...")

#             # 检测和绘制音符，并返回所有和弦的二进制表示
#             all_chords, totalcord = detect_and_draw_notes(img_path, model_note, model_region)

#             # 在数据库中搜索
#             matches = image_db.search_image(all_chords, tolerance=0)
#             ratios = {}
#             for img_id, regions_id in matches.items():
#                 match_cord = sum(regions_id.values())
#                 ratios[img_id] = match_cord / totalcord
            
#             # 获取匹配比率最高的前三个曲目
#             top_matches = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:3]
#             top_track_names = [image_db.track_info[image_id] for image_id, _ in top_matches]
            
#             if top_track_names:
#                 f.write(f"Image: {os.path.basename(img_path)}\n")
#                 for id, ratio in top_matches:
#                     f.write(f"Name: {image_db.track_info[id]}\t\tRatio: {ratio:.2f}\n")
#                 f.write("\n")
#             else:
#                 f.write(f"Image: {os.path.basename(img_path)}\n未找到匹配的曲目。\n\n")
            
#     print(f"Results saved to {result_file}")

# if __name__ == "__main__":
#     main()