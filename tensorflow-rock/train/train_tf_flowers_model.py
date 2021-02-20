from model.train_model import StarModel, TFFlowerModel

if __name__ == '__main__':
    TFFlowerModel(image_root="/WORK/datasset/flower_photos/train", val_image_root="/WORK/datasset/flower_photos/val", image_x=320,
              image_y=212) \
        .load(batch=10) \
        .train(steps_per_epoch=300,
               epochs=130) \
        .save("../save/model/star")
