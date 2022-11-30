import os

train_txt_path = os.path.join("data", "train.txt")
train_dir = os.path.join("data", "LANE")
valid_txt_path = os.path.join("data", "test.txt")
valid_dir = os.path.join("data", "LANE")

# def gen_txt(txt_path, img_dir):
#     f = open(txt_path, 'w')
    
#     for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
#         for sub_dir in s_dirs:
#             i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
#             print(i_dir)
#             label = i_dir.split('\\')[-1]
#             img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
#             for i in range(len(img_list)):
#                 if not img_list[i].endswith('png'):         # 若不是png文件，跳过
#                     continue
#                 #label = img_list[i].split('_')[0]
#                 img_path = os.path.join(i_dir, img_list[i])
#                 line = img_path + ' ' + label + '\n'
#                 f.write(line)
#     f.close()



def gen_txt(train_txt_path,valid_txt_path, img_dir):
    f1 = open(train_txt_path, 'w')
    f2 = open(valid_txt_path, 'w')
    
    step = 5

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            print(i_dir)
            label = i_dir.split('\\')[-1]
            img_list = os.listdir(i_dir)                    # 获取类别文件夹下所有png图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('png'):         # 若不是png文件，跳过
                    continue
                #label = img_list[i].split('_')[0]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                
                if i%step == 0:
                    f2.write(line)
                else:
                    f1.write(line)

    f1.close()
    f2.close()


if __name__ == '__main__':
    # gen_txt(train_txt_path, train_dir)
    # gen_txt(valid_txt_path, valid_dir)
    gen_txt(train_txt_path,valid_txt_path, valid_dir)
