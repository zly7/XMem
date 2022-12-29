# 这个主要用于部分的dataset已近被下载或者处理好的情况
import os
import gdown
import zipfile
import sys
sys.path.append("/home/songx_lab/cse12012530/cv_pro/XMem")
from scripts import resize_youtube



"""
YouTubeVOS dataset
"""
os.makedirs('../YouTube', exist_ok=True)
os.makedirs('../YouTube/all_frames', exist_ok=True)



print('Extracting YouTube datasets...')
with zipfile.ZipFile('../YouTube/train.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/')
with zipfile.ZipFile('../YouTube/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/')
with zipfile.ZipFile('../YouTube/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube/all_frames')

print('Cleaning up YouTubeVOS datasets...')
os.remove('../YouTube/train.zip')
os.remove('../YouTube/valid.zip')
os.remove('../YouTube/all_frames/valid.zip')

print('Resizing YouTubeVOS to 480p...')
resize_youtube.resize_all('../YouTube/train', '../YouTube/train_480p')

# YouTubeVOS 2018
os.makedirs('../YouTube2018', exist_ok=True)
os.makedirs('../YouTube2018/all_frames', exist_ok=True)


print('Extracting YouTube2018 datasets...')
with zipfile.ZipFile('../YouTube2018/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/')
with zipfile.ZipFile('../YouTube2018/all_frames/valid.zip', 'r') as zip_file:
    zip_file.extractall('../YouTube2018/all_frames')

print('Cleaning up YouTubeVOS2018 datasets...')
os.remove('../YouTube2018/valid.zip')
os.remove('../YouTube2018/all_frames/valid.zip')


"""
Long-Time Video dataset
"""
os.makedirs('../long_video_set', exist_ok=True)
print('Extracting long video dataset...')
with zipfile.ZipFile('../long_video_set/LongTimeVideo.zip', 'r') as zip_file:
    zip_file.extractall('../long_video_set/')
print('Cleaning up long video dataset...')
os.remove('../long_video_set/LongTimeVideo.zip')


print('Done.')