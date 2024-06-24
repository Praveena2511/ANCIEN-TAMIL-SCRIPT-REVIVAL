# %%
import joblib


print('model')

def model_generation():
    import os, glob
    import cv2
    import tensorflow as tf
    from tensorflow import keras
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras import datasets, layers, models

# %%
    CATEGORIES = []
    pics = []
    files = ['1 - Multipart','2 - Unknown']
    DATADIR = r'C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\data and model\images_categorised'
    for directoryfile in os.listdir(DATADIR):
       if(directoryfile in files):
        continue
       print(directoryfile)

       CATEGORIES.append(directoryfile)
    
    print(len(CATEGORIES))
    print(CATEGORIES)

# %%
    pic_dir = r'C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\data and model\images_categorised'
    pic_list = []


    filelist = glob.glob(os.path.join(pic_dir, '**\*.JPG'))

# %%
    print(filelist)
    print(len(filelist))

# %%
    cat_list = []
    for i in filelist:
#     print(i)
        a = i[168:170] # see to that the indices given are such a way that it properly takes the name of the category
        m = ""
        print(a)
        for i in a:
         if i.isnumeric():
            m += i
        cat_list.append(int(m))
    print(cat_list)  
    

# %%
    pic_list = []
    for i in filelist:
       picture = cv2.imread(i)
       picture_resized = cv2.resize(picture, (50, 50))
       print(picture_resized.shape)
       pic_list.append(picture_resized)

# %%
    parent = r'C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\data and model\augmented_images'
    for i in CATEGORIES:
       try:
         path = os.path.join(parent, i)
         os.makedirs(path)
         print("directiory created", path)
       except OSError:
         pass

# %%
#from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from tensorflow.keras.utils import array_to_img, img_to_array, load_img

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')


    for i in range(min(len(pic_list), len(cat_list))):
       x = pic_list[i]
       x = x.reshape((1,) + x.shape)
       lim = 0
    
       for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='C:/Users/praveena/Downloads/Tamil-Epigraphs-data-creation-and-recognition-main (1)/Tamil-Epigraphs-data-creation-and-recognition-main/data and model/augmented_images/'+str(cat_list[i]), save_prefix='', save_format='jpg'):
        #                                    make note of the slashes here... (/)... 
        
          lim += 1
        
          if lim > 10:
            break

# %%
    print(len(pic_list)) # see to that its 155 or else run the cell that has the following code

    '''pic_list = []
    for i in filelist:
    picture = cv2.imread(i)
    picture_resized = cv2.resize(picture, (50, 50))
    print(picture_resized.shape)
    pic_list.append(picture_resized)'''

# %%
    augmented_pic_dir = r'C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\project related data and model\augmented_images'
    filelist_aug = glob.glob(os.path.join(augmented_pic_dir, '**\*.JPG'))

# %%
    print(filelist_aug[:50])

# %%

    for i in filelist_aug:
#     print(i)
       a = i[182:184]  # as mentioned above... see to the value of index...
       m = ""
       print(a)
       for i in a:
         if i.isnumeric():
            m += i
       cat_list.append(int(m))
    print(cat_list) 
     

# %%

    for i in filelist_aug:
      picture = cv2.imread(i)
    
      print(picture)
      pic_list.append(picture)
    print(len(filelist))

# %%
    count_list = []
    for i in range(len(CATEGORIES)):
       print(cat_list.count(i))
       count_list.append(cat_list.count(i))

# %%
    from sklearn.model_selection import train_test_split

    X = np.array(pic_list)
    Y = np.array(cat_list)

    Y = Y.reshape(1852,1)
    print(X.shape)
    print(Y.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.80, random_state = 42)

# %%
    print(X_train.shape, Y_train.shape)

# %%
    X_train_scal = X_train/255.0
    X_test_scal  = X_test/255.0


# %%

    model = models.Sequential([
    
     layers.Conv2D(filters = 32, kernel_size = (4,4), activation = 'relu', input_shape = (50, 50, 3)),
     layers.MaxPooling2D((3,3)),
     layers.Dropout(0.3),
    
     layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
     layers.MaxPooling2D((3,3)),
#     layers.Dropout(0.1),
    
     layers.Flatten(),
     layers.Dense(200, activation = 'relu'),
     layers.Dense(28, activation = 'softmax')
    ])

    model.compile(
     optimizer = 'Adam',
     loss = 'sparse_categorical_crossentropy',
     metrics = ['accuracy']
    )
    model.fit(X_train_scal, Y_train, epochs = 50)

    

# %%
    model.evaluate(X_test, Y_test)

# %%
    Y_pred = model.predict(X_test)

# %%
    Y_pred_labels = [np.argmax(i) for i in Y_pred]

# %%
    from sklearn.metrics import classification_report
    print(classification_report(Y_test, Y_pred_labels))

# %%
    from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
    acc = accuracy_score(Y_test, Y_pred_labels)
    prec= precision_score(Y_test, Y_pred_labels, average = 'weighted')
    recall = recall_score(Y_test, Y_pred_labels, average = 'weighted')
    f1 = f1_score(Y_test, Y_pred_labels, average = 'weighted')

# %%
    print("accuracy  = ", acc)
    print("precision =", prec)
    print("recall    =", recall)
    print("f1_score  =", f1)

# %%
    
    while True:

      index_input = input("Enter the index of the test data you want to predict (or 'exit' to stop): ")
    
      if index_input.lower() == 'exit':
        break

        
      index = int(index_input)
      chosen_test_data = X_test[index]
        
      img1 = tf.expand_dims(chosen_test_data,0)
      print(img1.shape)
      y1 = model.predict(img1)
    
      y1_label = np.argmax(y1)
      print(y1_label)

      modern_pic1 = r'C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\project related data and model\Modern characters\{lab}\{lab2}.JPEG'.format(lab = y1_label, lab2 = y1_label)
      type(modern_pic1)
      modern_pic_1 = cv2.imread(modern_pic1)
      print(modern_pic_1)

      plt.imshow(modern_pic_1)

# Hide the axes
      plt.axis('off')

# Save the image as "output.jpg"
      plt.savefig("C:/Users/praveena/Downloads/Tamil-Epigraphs-data-creation-and-recognition-main (1)/Tamil-Epigraphs-data-creation-and-recognition-main/output.jpg", bbox_inches='tight', pad_inches=0)
      plt.close()



# %%
      
      img1 = tf.squeeze(img1)

      plt.imshow(img1.numpy())

      img1_numpy = img1.numpy()

      plt.imshow(img1_numpy)
# Hide the axes
      plt.axis('off')

# Save the image as "Input.jpg"
      plt.savefig("C:/Users/praveena/Downloads/Tamil-Epigraphs-data-creation-and-recognition-main (1)/Tamil-Epigraphs-data-creation-and-recognition-main/input.jpg", bbox_inches='tight', pad_inches=0)
      plt.close

    

# %%


