from model.train_model import StarModel

if __name__ == '__main__':
    StarModel(image_root="/WORK/datasset/star_imgs/train", val_image_root="/WORK/datasset/star_imgs/val", image_x=130,
              image_y=155) \
        .load(batch=10) \
        .train(steps_per_epoch=1000,
               epochs=80) \
        .save("../save/model/star")
