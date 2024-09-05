<h4 style="color:#1a96f6;"> <center> <b>

MIT License

Copyright (c) 2024 Felipe Braz da Silva e Pedro Wilson Silva de Sousa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</b> </center> </h4>

<div class="alert alert-info">
    
<h1 style="color:#1a96f6;"> <center> <b>
Projeto Final- Disciplina Técnicas de Prototipagem
</b> </center> </h1>
    
<h3 style="color:#1a96f6;"> <center> <b>
Projeto ECO
</b></center> </h3>
    
<h3 style="color:#1a96f6;"> <center> <b>
Prof. Dr. Moacy Pereira da Silva
</b></center> </h3>

<h3 style="color:#1a96f6;"> <center> <b>
Discentes: 

Felipe Braz da Silva

Pedro Wilson Silva de Sousa
</b></center> </h3>

<h3 style="color:#1a96f6;"> <center> <b>
Data: 16/19/2024
</b></center> </h3>


</div>

# Projeto ECO

#### *Objetivo do módulo*: Treinar um modelo de rede neural convulacional para identificação de materiais recicláveis.



##### Considerações:

A príncipio, o projeto visa identificar os materiais vidro e metal. Posteriormente, em pesquisas futuras consideramos extender a identificação a outros materiais comuns no processo de coleta seletiva. 

### Passo 1: Carregamento das imagens e divisão em treino e teste

As imagens foram adquiridas a partir do dataset Recyclable Solid Wast disponível na plataforma kaggle.

link para dataset: https://www.kaggle.com/datasets/hseyinsaidkoca/recyclable-solid-waste-dataset-on-5-background-co

#### 1.1 A ferramenta ImageDataGenerator 

A ferramenta ImageDataGenerator foi utilizada para dividir os dados em treinamento e teste. Utilizamos a proporção de 80% para treinamento e 20% para teste escolhidos de forma aleatória, padrão bem aceito em projetos de Machine Learning e Deep Learning.

Conjunto de treino: 424 imagens

Conjunto de teste: 161 imagens


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = 'C:/Users/felip/PycharmProjects/Notebooks IA/dataset'

# Diretórios de treinamento e teste
train_dir = dataset_path + '/train'
test_dir = dataset_path + '/test'

# Inicializar ImageDataGenerator para normalização
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Carregar o conjunto de treinamento
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150), 
    batch_size=32,
    class_mode='categorical' 
)

# Carregar o conjunto de teste
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

```

    Found 424 images belonging to 3 classes.
    Found 161 images belonging to 3 classes.
    


```python
print("Train generator: {} samples found.".format(train_generator.samples))
print("Test generator: {} samples found.".format(test_generator.samples))

```

    Train generator: 424 samples found.
    Test generator: 161 samples found.
    

### Passo 2: Construção do modelo de rede neural Convulacional

Utilizando como referência as ferramentas de modelagem de redes neurais mais atuais foi escolhida a utilização da biblioteca tensorflow com o módulo Keras.


*Modelo 1*: AlexNet, modelo de rede neural convulacional aplicado em Deep learning desenvolvido por Alex Krizhevsky, Ilya Sutskever e Geoff Hinton. Conhecido por vencer o ImageNet Large Scale Visual Recognition Challenge (ILSVRC) em 2012.

Mais informações:

https://vitalflux.com/different-types-of-cnn-architectures-explained-examples/

https://medium.com/@siddheshb008/alexnet-architecture-explained-b6240c528bd5]


*Modelo 2*: Modelo aleatório


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential(
    [
        Conv2D(96, kernel_size=(11, 11), strides=4,
               padding='valid', activation='relu',
               input_shape=(150, 150, 3),
               kernel_initializer='he_normal'),
    
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),

        Conv2D(256, kernel_size=(5, 5), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),
    
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),
        
        Conv2D(384, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),
    
        Conv2D(384, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),

        Conv2D(256, kernel_size=(3, 3), strides=1,
               padding='same', activation='relu',
               kernel_initializer='he_normal'),

        MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                     padding='valid', data_format=None),

        Flatten(),
    
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='relu'),
        Dense(3, activation='softmax')
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_4"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_18 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">35</span>, <span style="color: #00af00; text-decoration-color: #00af00">96</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">34,944</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">96</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">614,656</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_20 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">384</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">885,120</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_21 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">384</span>)      │     <span style="color: #00af00; text-decoration-color: #00af00">1,327,488</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">8</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │       <span style="color: #00af00; text-decoration-color: #00af00">884,992</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_16 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2304</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4096</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">9,441,280</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">4096</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">16,781,312</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1000</span>)           │     <span style="color: #00af00; text-decoration-color: #00af00">4,097,000</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">3,003</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,069,795</span> (129.97 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">34,069,795</span> (129.97 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model_2 = Sequential(
    [
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
    
        Dense(512, activation='relu'),
    
        Dropout(0.5),
    
        Dense(3, activation='softmax')    
    ]

)

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_2.summary()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_8"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_35 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">148</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_29 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">74</span>, <span style="color: #00af00; text-decoration-color: #00af00">74</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_36 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">72</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_30 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">36</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_37 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">34</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_31 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">17</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_38 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)              │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_32 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>) │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)      │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">6272</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_22 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │     <span style="color: #00af00; text-decoration-color: #00af00">3,211,776</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_23 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │         <span style="color: #00af00; text-decoration-color: #00af00">1,539</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,454,147</span> (13.18 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,454,147</span> (13.18 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



### Passo 3: Treinamento do modelo de rede neural Convulacional


```python
# Treinando o modelo 1: AlexNet
history_model_1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,  
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

```

    Epoch 1/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 271ms/step - accuracy: 0.8777 - loss: 0.3071 - val_accuracy: 0.9000 - val_loss: 0.2955
    Epoch 2/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9375 - loss: 0.2547 - val_accuracy: 0.0000e+00 - val_loss: 2.6851
    Epoch 3/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 256ms/step - accuracy: 0.8771 - loss: 0.3746 - val_accuracy: 0.9062 - val_loss: 0.2830
    Epoch 4/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.8125 - loss: 0.3871 - val_accuracy: 0.0000e+00 - val_loss: 0.9883
    Epoch 5/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 250ms/step - accuracy: 0.8867 - loss: 0.3093 - val_accuracy: 0.8938 - val_loss: 0.2276
    Epoch 6/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9062 - loss: 0.2448 - val_accuracy: 1.0000 - val_loss: 0.1784
    Epoch 7/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 256ms/step - accuracy: 0.8825 - loss: 0.3020 - val_accuracy: 0.9438 - val_loss: 0.1922
    Epoch 8/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9062 - loss: 0.1843 - val_accuracy: 1.0000 - val_loss: 0.1276
    Epoch 9/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 258ms/step - accuracy: 0.9259 - loss: 0.2255 - val_accuracy: 0.9125 - val_loss: 0.2736
    Epoch 10/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.8438 - loss: 0.3416 - val_accuracy: 1.0000 - val_loss: 0.3936
    Epoch 11/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 259ms/step - accuracy: 0.9209 - loss: 0.2359 - val_accuracy: 0.9563 - val_loss: 0.1452
    Epoch 12/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.9062 - loss: 0.1784 - val_accuracy: 1.0000 - val_loss: 1.7881e-05
    Epoch 13/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 256ms/step - accuracy: 0.9072 - loss: 0.2462 - val_accuracy: 0.9375 - val_loss: 0.2162
    Epoch 14/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9375 - loss: 0.1687 - val_accuracy: 1.0000 - val_loss: 9.6537e-04
    Epoch 15/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 258ms/step - accuracy: 0.8747 - loss: 0.3456 - val_accuracy: 0.9438 - val_loss: 0.2090
    Epoch 16/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 1.0000 - loss: 0.1156 - val_accuracy: 1.0000 - val_loss: 0.1686
    Epoch 17/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 267ms/step - accuracy: 0.9403 - loss: 0.1933 - val_accuracy: 0.9750 - val_loss: 0.1279
    Epoch 18/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.8750 - loss: 0.2919 - val_accuracy: 1.0000 - val_loss: 1.0729e-06
    Epoch 19/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 254ms/step - accuracy: 0.9571 - loss: 0.1540 - val_accuracy: 0.9312 - val_loss: 0.2005
    Epoch 20/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9375 - loss: 0.1846 - val_accuracy: 1.0000 - val_loss: 1.0729e-06
    Epoch 21/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 258ms/step - accuracy: 0.9533 - loss: 0.1624 - val_accuracy: 0.9563 - val_loss: 0.1452
    Epoch 22/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.9375 - loss: 0.1268 - val_accuracy: 1.0000 - val_loss: 0.1003
    Epoch 23/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 255ms/step - accuracy: 0.9448 - loss: 0.1687 - val_accuracy: 0.9750 - val_loss: 0.1138
    Epoch 24/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.9688 - loss: 0.1283 - val_accuracy: 1.0000 - val_loss: 0.1479
    Epoch 25/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 278ms/step - accuracy: 0.9233 - loss: 0.2072 - val_accuracy: 0.9563 - val_loss: 0.1439
    


```python
# Treinando o modelo 2: Aleatório
history_model_2 = model_2.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,  
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
```

    Epoch 1/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 173ms/step - accuracy: 0.3934 - loss: 1.1051 - val_accuracy: 0.4375 - val_loss: 1.0327
    Epoch 2/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.4688 - loss: 1.0660 - val_accuracy: 1.0000 - val_loss: 0.8166
    Epoch 3/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 156ms/step - accuracy: 0.4028 - loss: 1.0431 - val_accuracy: 0.7563 - val_loss: 0.8795
    Epoch 4/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.6562 - loss: 0.9135 - val_accuracy: 1.0000 - val_loss: 0.4847
    Epoch 5/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 166ms/step - accuracy: 0.6041 - loss: 0.8474 - val_accuracy: 0.8687 - val_loss: 0.3933
    Epoch 6/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.8750 - loss: 0.3751 - val_accuracy: 1.0000 - val_loss: 0.0031
    Epoch 7/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 160ms/step - accuracy: 0.7929 - loss: 0.4733 - val_accuracy: 0.8625 - val_loss: 0.3377
    Epoch 8/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7812 - loss: 0.4765 - val_accuracy: 1.0000 - val_loss: 0.3585
    Epoch 9/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 159ms/step - accuracy: 0.8893 - loss: 0.3284 - val_accuracy: 0.6250 - val_loss: 0.9436
    Epoch 10/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.6875 - loss: 1.1805 - val_accuracy: 1.0000 - val_loss: 0.1116
    Epoch 11/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 156ms/step - accuracy: 0.8146 - loss: 0.4236 - val_accuracy: 0.8938 - val_loss: 0.2308
    Epoch 12/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.9688 - loss: 0.1497 - val_accuracy: 1.0000 - val_loss: 0.0017
    Epoch 13/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 156ms/step - accuracy: 0.8965 - loss: 0.2697 - val_accuracy: 0.9125 - val_loss: 0.2065
    Epoch 14/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.9062 - loss: 0.2214 - val_accuracy: 1.0000 - val_loss: 0.6325
    Epoch 15/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 158ms/step - accuracy: 0.9081 - loss: 0.2536 - val_accuracy: 0.9625 - val_loss: 0.1081
    Epoch 16/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.9062 - loss: 0.2573 - val_accuracy: 1.0000 - val_loss: 0.0282
    Epoch 17/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 160ms/step - accuracy: 0.6991 - loss: 0.9493 - val_accuracy: 0.8687 - val_loss: 0.4545
    Epoch 18/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 0.7812 - loss: 0.5998 - val_accuracy: 0.0000e+00 - val_loss: 3.5644
    Epoch 19/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 164ms/step - accuracy: 0.8166 - loss: 0.5551 - val_accuracy: 0.8875 - val_loss: 0.3531
    Epoch 20/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step - accuracy: 1.0000 - loss: 0.0945 - val_accuracy: 1.0000 - val_loss: 0.2723
    Epoch 21/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 160ms/step - accuracy: 0.8658 - loss: 0.4289 - val_accuracy: 0.9250 - val_loss: 0.2290
    Epoch 22/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - accuracy: 0.8438 - loss: 0.4134 - val_accuracy: 1.0000 - val_loss: 0.0810
    Epoch 23/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 155ms/step - accuracy: 0.8790 - loss: 0.2868 - val_accuracy: 0.8687 - val_loss: 0.3236
    Epoch 24/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - accuracy: 0.9375 - loss: 0.2381 - val_accuracy: 1.0000 - val_loss: 0.0014
    Epoch 25/25
    [1m13/13[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 164ms/step - accuracy: 0.9026 - loss: 0.2530 - val_accuracy: 0.9375 - val_loss: 0.2050
    

### Passo 4: Avaliação do modelo de rede neural Convulacional


```python
# Avaliando o modelo 1
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Accuracy: {:.2f}".format(test_accuracy * 100))

```

    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - accuracy: 0.9612 - loss: 0.1399
    Test Accuracy: 95.65
    


```python
# Avaliando o modelo 2
test_loss, test_accuracy = model_2.evaluate(test_generator)
print("Test Accuracy: {:.2f}".format(test_accuracy * 100))

```

    [1m6/6[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - accuracy: 0.9305 - loss: 0.2293
    Test Accuracy: 93.79
    

De acordo com os resultados obtidos foi escolhido o modelo AlexNet com desempenho de 95.6 % de Acurácia. Para o intuito do projeto podemos considerar um bom desempenho, não deixando de considerar a possibilidade de realizar pesquisas futuras para refinamento e escolha do modelo na rede neural convulacional adequado e execução de baterias de teste para devida validação do modelo.

**Observação**: o modelo AlexNet foi projetado para ser usado com conjuntos de dados de imagens em larga escala, mas para testar o comportamento obtido com nosso dataset decidimos testar.

### Passo 5: Salvar o modelo


```python
# Salvando o modelo
model.save('alexNet.h5')
model_2.save('cnn1.h5')
```

    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. 
    
