# Import everything needed to edit video clips
from moviepy.editor import *
import pandas as pd
import os

A1_PATH="./data/A1"
A1_CLIP_PATH="./data/A1_clip"
NUM_CLASSES=16

def cut_video(clip1, time_start, time_end, path_video):
    # clip1 = VideoFileClip("test_phone.mp4").subclip(5, 18)
    clip1 = clip1.subclip(time_start, time_end)
    # getting width and height of clip 1
    w1 = clip1.w
    h1 = clip1.h
    
    print("Width x Height of clip 1 : ", end = " ")
    print(str(w1) + " x ", str(h1))
    
    print("---------------------------------------")
    
    # # resizing video downsize 50 %
    clip2 = clip1.resize((512, 512))

    # getting width and height of clip 1
    w2 = clip2.w
    h2 = clip2.h
    
    print("Width x Height of clip 2 : ", end = " ")
    print(str(w2) + " x ", str(h2))
    
    print("---------------------------------------")
    clip2.write_videofile(path_video)

#create folder data
if not os.path.isdir(A1_CLIP_PATH):
    os.makedirs(A1_CLIP_PATH)
else: 
    print("folder already exists.")

for i in range(NUM_CLASSES):
    data_dir = '{}/{}'.format(A1_CLIP_PATH, str(i))
    CHECK_FOLDER = os.path.isdir(data_dir)
    if not CHECK_FOLDER:
        os.makedirs(data_dir)
    else:
        print(data_dir, "folder already exists.")
    print(i)

data_list = []

for folder_name in os.listdir(A1_PATH):
    if not folder_name.startswith("user_id_"):
        continue
    # print(folder_name)
    path_folder = '{}/{}'.format(A1_PATH,folder_name)
    path_csv = '{}/{}.csv'.format(path_folder, folder_name)
    print(path_folder, path_csv)
    df = pd.read_csv(path_csv)
    filename_lst = list(df['Filename'].values)
    label_lst = list(df['Label (Primary)'].values)
    start_time_lst = list(df['Start Time'].values)
    end_time_lst = list(df['End Time'].values)

    prev_file_name = ''
    file_name = ''
    count = 0
    for i in range(len(df)):
        if filename_lst[i] !=' ' and str(filename_lst[i]) != 'nan' and len(str(filename_lst[i]))>16:
            file_name = filename_lst[i].replace(' ', '')
            file_name_parts = file_name.split("_")
            file_name_parts.insert(-1, "NoAudio")
            file_name = '_'.join(file_name_parts)
            file_name = file_name.replace("User", "user")
            file_name = file_name.replace("Rearview", "Rear_view")
            print('file_name', filename_lst[i], file_name)
            # print(count)
            count = 0

        if label_lst[i] == 'NA':
            continue

        if file_name != prev_file_name:
            video_path = os.path.join(path_folder, file_name+".MP4")
            clip = VideoFileClip(video_path)
            clip_duration = int(clip.duration)
            prev_file_name = file_name
        if label_lst[i].strip(" ").lstrip("Class") == "":
            continue
        clip_label = int(label_lst[i].strip(" ").lstrip("Class"))
        ftr = [3600,60,1]
        
        # segment distract clip
        time_start = sum([a*b for a,b in zip(ftr, map(int,start_time_lst[i].split(':')))])
        time_end = sum([a*b for a,b in zip(ftr, map(int,end_time_lst[i].split(':')))])
        if time_start > clip_duration or time_end > clip_duration:
            if time_start>900 or time_end>900:
                print(start_time_lst[i], end_time_lst[i])
                print("Annotation Error", video_path)
                exit(-1)
            else:
                time_end = clip_duration
        clip_path = '{}/{}/{}_{}_{}.MP4'.format(A1_CLIP_PATH, clip_label, file_name, time_start, time_end)
        data_list.append([clip_path, clip_label])

        if not os.path.exists(clip_path):
            print("process {}".format(clip_path))
            cut_video(clip, time_start, time_end, clip_path)
        else:
            print("Already process {}".format(clip_path))
        if i == (len(df) - 1):
            print("Finished file {}".format(path_csv))
            break
        #Segment normal clip
        time_start = sum([a*b for a,b in zip(ftr, map(int,end_time_lst[i].split(':')))])
        time_end = sum([a*b for a,b in zip(ftr, map(int,start_time_lst[i+1].split(':')))])   
        if time_start > clip_duration or time_end > clip_duration:
            if time_start>900 or time_end>900:
                print(start_time_lst[i], end_time_lst[i])
                print("Annotation Error", video_path)
                exit(-1)
            else:
                time_end = clip_duration
        time_end = time_end if time_end > time_start else clip_duration

        if abs(time_end-time_start)>200 or time_start >= time_end:
            continue
        clip_path = '{}/{}/{}_{}_{}.MP4'.format(A1_CLIP_PATH, 0, file_name, time_start, time_end)
        data_list.append([clip_path, 0])

        if not os.path.exists(clip_path):
            print("process {}".format(clip_path))
            print("Clip duration", clip_duration, time_start, time_start)
            time_end = min(time_end, clip_duration)
            cut_video(clip, time_start, time_end, clip_path)
        else:
            print("Already process {}".format(clip_path))
        count +=1
            # break
        print(file_name)




# with open(os.path.join(A1_CLIP_PATH, "data_clean.txt"), "w+") as fp:
#     for data_path, label in data_list:
#         fp.writelines("{} {}\n".format(data_path, label))