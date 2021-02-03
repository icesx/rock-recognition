from model.train_model import RockModel, Mnist

if __name__ == '__main__':
    Mnist(image_root="/WORK/datasset/mnist/train",test_image_root="/WORK/datasset/mnist/test") \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=50) \
        .save("../save/model/mnist")
