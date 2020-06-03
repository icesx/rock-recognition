<<<<<<< HEAD
<<<<<<< HEAD
from model.train_model import RockModel
=======
from model.my_model import RockModel
>>>>>>> 1d6ecd4... basemodeloperate
=======
from model.train_model import RockModel
>>>>>>> 410dda6... remove model step-1

if __name__ == '__main__':
    RockModel(image_root="/WORK/datasset/rock_imgs_train", image_x=128, image_y=128) \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=300) \
        .save("../save/model/rock")
