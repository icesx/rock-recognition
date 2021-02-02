from model.train_model import RockModel, Mnist

if __name__ == '__main__':
    Mnist(image_root="/WORK/datasset/fashion_mnist/train") \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=300) \
        .save("../save/model/fashion_mnist")
