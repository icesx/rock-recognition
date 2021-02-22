from model.train_model import TFFlowerModel

if __name__ == '__main__':
    TFFlowerModel(image_root="/WORK/datasset/flower_photos/train", val_image_root="/WORK/datasset/flower_photos/val",
                  image_x=220,
                  image_y=212) \
        .load(batch=50) \
        .train(steps_per_epoch=400,
               epochs=80) \
        .save("../save/model/star")
