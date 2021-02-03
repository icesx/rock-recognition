from model.train_model import RockModel

if __name__ == '__main__':
    RockModel(image_root="/WORK/datasset/rock_imgs/train",val_image_root="/WORK/datasset/rock_imgs/val", image_x=128, image_y=128) \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=300) \
        .save("../save/model/rock")
