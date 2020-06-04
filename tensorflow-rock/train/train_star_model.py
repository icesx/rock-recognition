from model.train_model import StarModel

if __name__ == '__main__':
    StarModel(train_image_root="/WORK/datasset/star_imgs_train",
              test_image_root="/WORK/datasset/star_imgs_test",
              image_x=130, image_y=155) \
        .load(batch=20) \
        .train(steps_per_epoch=300,
               epochs=10) \
        .save("../save/model/star")
