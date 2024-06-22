<div align='center'>

# Instituto Politécnico de Leiria 


<img src="IPL_LOGO.png"   width="100"/>

<br>

# Relatório de Inteligência Artificial
<br>




# Engenharia informática


<br>
<br>

 <b>Carlos Vinagre - 2212552</b>
 <br>
 <b>Alexandre Jesus - 2211037</b>

</div>


---

## Introdução

O nosso projeto consiste na tarefa de classificação de imagens. Para realizar esta tarefa, utilizamos as seguintes bibliotecas:

- **Keras:** É uma biblioteca de deep learning de alto nivel.É usada para a criação de modelos de deep learning e é executada sobre a plataforma <b>Tensorflow</b>.
<br>


- **Matplotlib:** É uma biblioteca de visualização de dados no nosso projeto sera usado para visualizar imagens e graficos de progresso.
<br>

- **TensorFlow:** TensorFlow é uma biblioteca de codigo aberto para machine learning pode ser executada tanto usando cpu como gpu ou tpu(Tensor Processing Units)

<br>

- **NumPy:** É uma biblioteca matematica onde da para criar estruturas multidimensioais, tambem oferece varias funções matematicas uteis para a utilização neste projeto


---
## Código

`from google.colab import drive
drive.mount('/content/drive/')`

- Estas linhas de código serve para montar um projeto que esta no google drive ao Google Colab.

---

` import os, shutil
train_dir = '/content/drive/MyDrive/AI/Projeto_AI/train'
validation_dir = '/content/drive/MyDrive/AI/Projeto_AI/train5'
test_dir = '/content/drive/MyDrive/AI/Projeto_AI/test'
... `

- Estas linhas de código serve para dizer onde esta o caminho das imagens do treino , validação e testes, tambem conta o numero das mesmas.

---

`
from keras.utils import image_dataset_from_directory
IMG_SIZE = 64
train_dataset = image_dataset_from_directory(
train_dir,
image_size=(IMG_SIZE, IMG_SIZE),
batch_size=32)
validation_dataset = image_dataset_from_directory(
validation_dir,
image_size=(IMG_SIZE, IMG_SIZE),
batch_size=32)
test_dataset = image_dataset_from_directory(
test_dir,
image_size=(IMG_SIZE, IMG_SIZE),
batch_size=32)
`
- Estas linhas de código serve para definir o tamanho das imagens cria o conjunto de treino, validação e teste.

---

`
for data_batch, labels_batch in train_dataset:
  print('data batch shape:', data_batch.shape)
  print('labels batch shape:', labels_batch.shape)
  break
  `
  - Esta parte do código cria um loop em que as variaveis recebem um conjunto de dados e onde tambem imprime as dimensões dos mesmos.

  ---

  `
  import matplotlib.pyplot as plt
for data_batch, _ in train_dataset.take(1):
  for i in range(5):
    plt.imshow(data_batch[i].numpy().astype("uint8"))
    plt.show()
`
- Este código vai buscar a biblioteca matplotlib e usa-a para mostrar 5 imagens.

---

`
from tensorflow import keras
from keras import layers
from keras import models

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
`

- O código anteriormente mencionado cria uma rede neuronal convolucional (CNN) composta por varias camadas e com varios neuronios onde na camada densa de saida com um único neurônio.

---

`
def get_model():
    import tensorflow as tf
    model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])
    return model
`

- O código configura o modelo para treinamento com a função de perda e usa o otimizador RSMprop.

`
history = model.fit(
train_dataset,
epochs=10,
validation_data=validation_dataset)
`

- Esta função treina o modelo por 10 epocas usando um conjunto de dados e avalia o depois de cada época.

`
from tensorflow import keras
model.save("/content/drive/MyDrive/AI/Projeto_AI.h5")
print("Saved model to disk")
model = keras.models.load_model('/content/drive/MyDrive/AI/Projeto_AI.h5')
#--------------------------
val_loss, val_acc = model.evaluate(validation_dataset)
print('val_acc:', val_acc)
`

-Esta função guarda o modelo treinado , carrega o mesmo e avalia-o e imprime a acurácia.

`
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
`

## Código do 2º Modelo

1. Imports usados para fine-tuning.
(Usámos o modelo pré treinado VGG16 do keras)

`
from tensorflow.keras.applications import VGG16
`
2. Buscar pastas com as imagens

`
import os
import tensorflow as tf
train_dir = '/content/drive/MyDrive/AI/Projeto_AI/train'
validation_dir = '/content/drive/MyDrive/AI/Projeto_AI/train5'
test_dir = '/content/drive/MyDrive/AI/Projeto_AI/test'
`
3. Fazer o preprocessamento dos dados.

`
IMG_SIZE = 150
BATCH_SIZE = 32
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
`

4. Compilação do modelo neural
(Neste 2º modelo usámos um otimizador distinto, o otimizador adam em vez de RMSprop e usamos a função de loss sparse_categorical_crossentropy em vez de binary_crossentropy)
`
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
`
5. Treinar o modelo

`
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')
model.save('/content/drive/MyDrive/AI/Projeto_AI/image_classifier_model.h5')
`
6. Mostrar o gráfico de progresso
(Mostra os valores de accuracy e valores de loss)
`
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Accuracy')
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
`

# Código do 2º Modelo modificado para usar data augmentation

3. Fazer o preprocessamento dos dados com a utilização de uma função de data augmentation
(Usámos as funções de augmentação RandomFlip e RandomRotation para aumentar a diversidade dos conjuntos de treino)
`
IMG_SIZE = 150
BATCH_SIZE = 32
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
).map(lambda x, y: (data_augmentation(x, training=True), y))
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
`

# Código do 2º Modelo modificado para usar fine-tuning

3. Criação do modelo neural com fine-tuning.
(AUTOTUNE utilizado para melhorar o transfer learning, utilização do modelo pré treinado VGG16, congelou-se o modelo base e adicionou-se camadas de classificação em cima do modelo)
`
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_dataset.class_names), activation='softmax')
])
`

4. Compilação do modelo neural com fine-tuning.
(Descongelamento das camadas base do modelo, fine-tune desde a camada 10, congelamento de todas as camadas antes da 10, compilação od modelo com um menor learning rate)
`
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 10
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
`              

5. Treinar o modelo com fine-tuning.
(Treino com 10 épocas em que se avalia o modelo e se guarda o modelo avaliado num ficheiro)
`
history_fine = model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset
)
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy (Fine-Tuning): {test_acc}')
model.save('/content/drive/MyDrive/AI/Projeto_AI/image_classifier_model_fine_tuning.h5')
`


























