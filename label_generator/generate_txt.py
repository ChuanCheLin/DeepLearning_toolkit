import os
# write txt file
txt_dir = "/home/eric/few-shot-object-detection/output/unlabel.txt"
file = open(txt_dir, 'w')

path = "/home/eric/few-shot-object-detection/output/unlabel_jpg/"
dirs = os.listdir(path)

for name in dirs:
    file.write(name)
    file.write('\n')

file.close()