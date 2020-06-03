<<<<<<< HEAD
from model.train_model import RockModel
=======
from model.my_model import RockModel
>>>>>>> 1d6ecd4... basemodeloperate

if __name__ == '__main__':
    RockModel(image_root="/WORK/datasset/rock_imgs_train2", image_x=128, image_y=128) \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=100) \
        .save("../save/rock")
