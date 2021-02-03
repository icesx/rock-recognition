from model.train_model import StarModel, Flower102Model

if __name__ == '__main__':
    Flower102Model(image_root="/WORK/datasset/102flowers/train", val_image_root="/WORK/datasset/102flowers/val",
                   image_x=256,
                   image_y=256) \
        .load(batch=10) \
        .train(steps_per_epoch=100,
               epochs=500) \
        .save("../save/model/star")
