model = build_unet(image_patch_size, len(color_mapping))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=[X_valid, y_valid], epochs=100)

model.save("models/model_sem_seg_dubai.h5")
