import os
xml_file = './Annotations/'
img_file = './images/'
test_file = './test_img/'
def compare_xml_jpg():
    for i in os.listdir(img_file):
        flag = 0
        for j in os.listdir(xml_file):
            if i.split('.')[0] in j.split('.')[0]:
                flag = 1
        if flag == 0:
            print(i)
            os.remove(img_file + i)


def del_some():
    pic_name = []
    total = 0
    for i in os.listdir(test_file):
        flag = 0
        if 'mix_UsingPhone' in i:
            for j in os.listdir(test_file):
                if j == i:
                    continue
                elif i.split('_')[-1].split('.')[0] == j.split('_')[-1].split('.')[0]:
                    print(i.split('_')[-1].split('.')[0], j.split('_')[-1].split('.')[0])
                    flag = 1
        if flag == 0:
            total += 1
            pic_name.append(i)

    print(pic_name)
    print(total)




    # for i in pic_name:
    #     try:
    #         os.remove('./images/'+ i)
    #     except:
    #         print('./images/'+ i)
    #     try:
    #         os.remove('./Annotations/' + i)
    #     except:
    #         print('./Annotations/' + i)
    #     try:
    #         os.remove('./mix_img/' + i)
    #     except:
    #         print('./mix_img/' + i)
    #     try:
    #         os.remove('./test_img/' + i)
    #     except:
    #         print('./test_img/' + i)


if __name__ == '__main__':
    compare_xml_jpg()
    # del_some()
