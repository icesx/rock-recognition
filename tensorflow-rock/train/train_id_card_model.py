from model.train_model import IDCardModel

if __name__ == '__main__':
    IDCardModel(image_root="/WORK/datasset/id_card/train", val_image_root="/WORK/datasset/id_card/val", image_y=256,
              image_x=32) \
        .load(batch=10) \
        .train(steps_per_epoch=1000,
               epochs=80) \
        .save("../save/model/star")
