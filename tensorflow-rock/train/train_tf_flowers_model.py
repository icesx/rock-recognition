from model.train_model import TFFlowerModel

if __name__ == '__main__':
    TFFlowerModel(image_root="/WORK/datasset/flower_photos/train", val_image_root="/WORK/datasset/flower_photos/val",
                  image_x=220,
                  image_y=212) \
        .load(batch=10) \
        .train(steps_per_epoch=1000,
               epochs=100) \
        .save("../save/model/star")
