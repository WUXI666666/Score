import glob
import os
import cv2
import torch
from searchScore import ImageDatabase,detect_and_draw_notes,separate_hands,detect_regions,group_chords,chords_to_binary


model_note = torch.jit.load('./models/score_note_320.torchscript.pt')
model_region = torch.jit.load('./models/region.torchscript.pt')
model_note.eval()
model_region.eval()
def create_database(image_paths, database_path, model_note, model_region, note_input_size=320, region_input_size=640):
    image_db = ImageDatabase()
    
    # 如果 image_paths 是文件夹路径，则获取该文件夹中的所有图像路径
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = glob.glob(os.path.join(image_paths, "*.jpg")) + \
                      glob.glob(os.path.join(image_paths, "*.png")) + \
                      glob.glob(os.path.join(image_paths, "*.jpeg"))
    elif not isinstance(image_paths, list):
        raise ValueError("image_paths should be a list of image paths or a directory containing images.")

    for img_path in image_paths:
        print(f"Processing {img_path}...")
        # 检测和绘制音符，并返回所有和弦的二进制表示
        all_chords,_ = detect_and_draw_notes(img_path, model_note, model_region, note_input_size, region_input_size)
        
        # 获取曲目名称
        track_name = os.path.basename(img_path).split('.')[0]
        
        # 添加到数据库
        image_db.add_image(track_name, all_chords, track_name)
    
    # 保存数据库
    image_db.save(database_path)
    print(f"Database saved to {database_path}")


if __name__ == "__main__":
    image_paths = './db'  
    database_path = './music_db.pkl'
    create_database(image_paths, database_path, model_note, model_region)



