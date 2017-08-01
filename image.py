from PIL import Image
import numpy as np
import params
import matplotlib.pyplot as plt

def image_show(img):
    plt.figure()
    plt.imshow(img)
    plt.show()

def ratio(img):
    return img.shape[1] / float(img.shape[0])

def split(img, piece_size = params.ORIGIN_SIZE / 51):
    assert(img.shape[0] % piece_size[0] == 0)
    assert(img.shape[1] % piece_size[1] == 0)
    h, w, _ = img.shape
    h_step = piece_size[0]
    w_step = piece_size[1]
    pieces = []
    for i in range(0, h, h_step):
        pieces.append([])
        for j in range(0, w , w_step):
            pieces[-1].append(img[i:i+h_step, j:j+w_step, :])
    return pieces

def aggregate(pieces):
    h_step, w_step, _ = pieces[0][0].shape
    h = len(pieces) * h_step
    w = len(pieces[0]) * w_step
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(0, h, h_step):
        for j in range(0, w, w_step):
            img[i:i+h_step, j:j+w_step, :] = pieces[i/h_step][j/w_step]
    return img

#def shrink(img, size):
#    #assert(img.shape[0] % size[0] == 0)
#    #assert(img.shape[1] % size[1] == 0)
#    assert(new_size[1] / float(new_size[0]) == ratio(img))
#    times = img.shape[0] / size[0]
#    return img[::times, ::times]

def scale(img, new_size):
    assert(new_size[1] / float(new_size[0]) == ratio(img))
    times = img.shape[0] / float(new_size[0])
    img_new = np.zeros((new_size[0], new_size[1], 3), dtype=np.uint8)
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            i_ = int(i * times)
            j_ = int(j * times)
            img_new[i][j] = img[i_][j_]
    return img_new

def rgb(img):
    return img.reshape(-1, 3).mean(axis=0)

if __name__ == '__main__':
    img = np.array(Image.open('./example.jpg'))
    #pieces = split(img)
    #img1 = aggregate(pieces)
    img_s = shrink(img, params.ORIGIN_SIZE / 51)
    img_new = scale(img_s, params.ORIGIN_SIZE / 17)
    image_show(img_new)
    image_show(img_s)
