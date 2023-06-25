import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Veri kümesini yükleyin ve ön işleme yapın
# (veri yükleme ve ön işleme kodu buraya gelebilir)

# Derin öğrenme modelini oluşturun
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Modeli derleyin
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitin
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Modeli değerlendirin
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
