<<<<<<< HEAD
from model.train_model import StarModel
=======
from model.my_model import StarModel
>>>>>>> 1d6ecd4... basemodeloperate

if __name__ == '__main__':
    StarModel(image_root="/WORK/datasset/star_imgs", image_x=128, image_y=128) \
        .load(batch=10) \
        .train(steps_per_epoch=100,
               epochs=30) \
        .save("../save/star")
