from model.train_model import RockModel

if __name__ == '__main__':
    RockModel(train_image_root="/WORK/datasset/rock_imgs_train", image_x=128, image_y=128) \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=300) \
        .save("../save/model/rock")
