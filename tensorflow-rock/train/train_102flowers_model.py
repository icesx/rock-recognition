from model.train_model import StarModel, Flower102Model

if __name__ == '__main__':
    Flower102Model(image_root="/WORK/datasset/102flowers/train", val_image_root="/WORK/datasset/102flowers/val",
                   image_x=256,
                   image_y=256) \
        .load(batch=20) \
        .train(steps_per_epoch=1000,
               epochs=200) \
        .save("../save/model/flowers102")
