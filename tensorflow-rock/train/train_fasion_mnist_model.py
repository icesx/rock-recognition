from model.train_model import RockModel, Mnist, FashionMnist

if __name__ == '__main__':
    FashionMnist(image_root="/WORK/datasset/fashion_mnist/train",val_image_root="/WORK/datasset/fashion_mnist/test") \
        .load(batch=15) \
        .train(steps_per_epoch=150,
               epochs=100) \
        .save("../save/model/fashion_mnist")
