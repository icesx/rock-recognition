from model.train_model import Cifar10

if __name__ == '__main__':
    Cifar10(image_root="/WORK/datasset/cifar10/train", val_image_root="/WORK/datasset/cifar10/val") \
        .load(batch=15) \
        .train(steps_per_epoch=250,
               epochs=300) \
        .save("../save/model/cifar10")
