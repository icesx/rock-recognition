from model.train_model import RockModel, Mnist

if __name__ == '__main__':
    Mnist(image_root="/WORK/datasset/mnist/train",val_image_root="/WORK/datasset/mnist/test") \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=100) \
        .save("../save/model/mnist")
