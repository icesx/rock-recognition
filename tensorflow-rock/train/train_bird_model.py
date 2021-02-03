from model.train_model import BirdModel

if __name__ == '__main__':
    BirdModel(image_root="/WORK/datasset/bird/train", val_image_root="/WORK/datasset/bird/val") \
        .load(batch=15) \
        .train(steps_per_epoch=250,
               epochs=200) \
        .save("../save/model/fashion_mnist")
