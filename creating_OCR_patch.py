import pandas as pd
import tifffile
import imagecodecs
import openpyxl
import matplotlib.pyplot as plt
import os
from PIL import Image

OCR_information_path = '/Users/yongchanchun/Desktop/MacBook_Pro_Desktop/graduate_school/SDS_project/OCR_information'
OCR_dataset_path = '/Users/yongchanchun/Desktop/MacBook_Pro_Desktop/graduate_school/SDS_project/Dataset_OCR'

for dir1, dir2 in zip(os.listdir(OCR_information_path), os.listdir(OCR_dataset_path)):
    print(dir1,'\n', dir2)
    info_table = openpyxl.load_workbook(os.path.join(OCR_information_path, dir1))
    ws = info_table['Sheet']
    data = ws.values
    columns = next(data)[0:]
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index('컬럼명')
    diagnose_start = df['추출 Data']['진료시작일자위치값'].split('^')
    diagnose_end = df['추출 Data']['진료종료일자위치값'].split('^')
    img = Image.open(os.path.join(OCR_dataset_path, dir2))
    img2 = img.resize((1429, 2048))
    x1, y1 = int(diagnose_end[0]), int(diagnose_end[1])
    x2, y2 = int(diagnose_end[2]), int(diagnose_end[3])
    img3 = img2.crop((x1, y1, x2, y2))
    img3.show()


original_path_cropped_all = '/Users/yongchanchun/Desktop/MacBook_Pro_Desktop/graduate_school/SDS_project/Dataset/original'
modified_path = '/Users/yongchanchun/Desktop/MacBook_Pro_Desktop/graduate_school/SDS_project/Dataset/modified'
img = Image.open(modified_path + '/오려붙이기/모두/위조_오려붙이기_진료기간,환자명_진료기간(월)_20220609_161623.jpg')
img_original = Image.open(original_path_cropped_all + '/20220609_161623.jpg')

# rotatation degree, x, y, w, h 설정
rot_deg = 357
x = 1715
y = 1020
w = 216
h = 36

img_modified_rotated = img.rotate(rot_deg)
img_original_rotated = img_original.rotate(rot_deg)
img_modified_rotated = img_modified_rotated.crop((x, y, x + w, y + h))
img_original_rotated = img_original_rotated.crop((x, y, x + w, y + h))

plt.subplot(2,1,1)
plt.imshow(img_modified_rotated)

plt.subplot(2,1,2)
plt.imshow(img_original_rotated)
plt.show()

img_modified_rotated.save(modified_path + '/오려붙이기/모두/위조_오려붙이기_진료기간,환자명_진료기간(월)_20220609_161623_patch.png', 'PNG')
img_original_rotated.save(original_path_cropped_all + '/20220609_161623_patch.png', 'PNG')
