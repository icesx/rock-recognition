from model.train_model import StarModel

if __name__ == '__main__':
    StarModel(train_image_root="/WORK/datasset/star_imgs_train",
              test_image_root="/WORK/datasset/star_imgs_test",
              image_x=128, image_y=128) \
        .load(batch=21) \
        .train(steps_per_epoch=300,
               epochs=20,
               validation_steps=21) \
        .save("../save/model/star")
