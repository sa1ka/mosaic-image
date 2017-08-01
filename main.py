import numpy as np
import matplotlib.pyplot as plt
import image
import pickle
import params
from PIL import Image
import glob

class PicLab(object):
    def __init__(self, shrink_size=None, lab_path=None, ratio_filter=1.5):
        pics = glob.glob(lab_path + '/*.JPG') if lab_path is not None else []
        pics += glob.glob(lab_path + '/*.jpg') if lab_path is not None else []
        self.pics = []
        for pic in pics:
            print pic
            img = np.array(Image.open(pic))
            r = image.ratio(img)
            if ratio_filter is not None and ratio_filter != r:
                continue
            self.pics.append(image.scale(img, shrink_size))
        self.rgbs = map(image.rgb, self.pics)

    def similar_img(self, img, num=20):
        rgb = image.rgb(img)
        def norm(n):
            return np.linalg.norm((n - rgb), ord=1)
        dis = map(norm, self.rgbs)
        sim = np.argsort(dis)[:num]
        return sim

    def random_similar_img(self, img, num=20):
        imgs = self.similar_img(img, num)
        return self.pics[np.random.choice(imgs)]

    def save(self, path='pics.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.pics, f)

    def load(self, path='pics.pkl'):
        with open(path, 'rb') as f:
            self.pics = pickle.load(f)
            self.rgbs = map(image.rgb, self.pics)

if __name__ == '__main__':
    img = np.array(Image.open('./example1.jpg'))
    img_new = image.scale(img, params.ORIGIN_SIZE)

    pieces_size = params.ORIGIN_SIZE / 17
    if False:
        pic_lab = PicLab(lab_path = './pic_lab', shrink_size = pieces_size * 3)
        pic_lab.save()
    else:
        pic_lab = PicLab()
        pic_lab.load()

    pieces = image.split(img_new, pieces_size)

    sim = []
    for i, l in enumerate(pieces):
        sim.append([])
        for j, p in enumerate(l):
            sim[-1].append(pic_lab.random_similar_img(p))

    img_agg = image.aggregate(sim)
    img_new = image.scale(img_new, params.ORIGIN_SIZE * 3)
    alpha = .7
    img_final = np.clip(img_agg * (1 - alpha) + img_new * .9, 0, 255).astype(np.uint8)
    #image.image_show(img_final)
    f, ax = plt.subplots(2)
    ax[0].imshow(img_final)
    ax[1].imshow(img_new)
    plt.show()
    input()


