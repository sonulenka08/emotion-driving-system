# Importing all the required Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers # Not directly used, layers are imported individually
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import load_img, img_to_array # Using this for imageshow
import numpy as np
import pandas as pd
import os
import itertools

# Define base paths
base_data_path = r"C:\Users\HP\Desktop\FER2013"
train_path = os.path.join(base_data_path, "train")
val_path = os.path.join(base_data_path, "valid")
# test_path = os.path.join(base_data_path, "test") # Defined, though not used in this script for training/validation

base_output_path = r"C:\Users\HP\Desktop\desk"
os.makedirs(base_output_path, exist_ok=True) # Create output directory if it doesn't exist

# Define output file paths
# MODIFIED HERE: Changed file extension for model weights checkpoint
model_weights_path = os.path.join(base_output_path, "model_best.weights.h5") # for checkpoint
model_plot_path = os.path.join(base_output_path, "emotion_model_architecture.png")
final_model_save_path = os.path.join(base_output_path, "CNN_Model_emotion_trained.h5") # for final trained model
history_csv_path = os.path.join(base_output_path, "training_history.csv")
predictions_npy_path = os.path.join(base_output_path, "validation_predictions.npy")
confusion_matrix_plot_path = os.path.join(base_output_path, "normalized_confusion_matrix.png")
accuracy_report_path = os.path.join(base_output_path, "accuracy_classification_report.txt")
training_plots_path = os.path.join(base_output_path, "training_performance_plots.png")


# --- Data Exploration (Optional, but good to keep) ---
if os.path.exists(train_path) and os.path.exists(val_path):
    print(f"Train path: {train_path}")
    print(f"Validation path: {val_path}")
    print(f"Output path: {base_output_path}\n")

    # data categories
    categories = []
    if os.path.exists(train_path):
        categories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    print("Categories found:", categories)

    # Training images
    total_train_images = 0
    for dir_ in categories: # Iterate through known categories
        category_dir_path = os.path.join(train_path, dir_)
        # if os.path.isdir(category_dir_path): # Already ensured dir_ is a directory
        count = len([name for name in os.listdir(category_dir_path) if os.path.isfile(os.path.join(category_dir_path, name))])
        total_train_images += count
        print(f"{dir_} has {count} number of training images")
    print(f"\nTotal train images are {total_train_images}")

    # Validation Images
    total_validation_images = 0
    val_categories = []
    if os.path.exists(val_path):
        val_categories = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]

    for dir_ in val_categories:
        category_dir_path = os.path.join(val_path, dir_)
        # if os.path.isdir(category_dir_path): # Already ensured dir_ is a directory
        count = len([name for name in os.listdir(category_dir_path) if os.path.isfile(os.path.join(category_dir_path, name))])
        total_validation_images += count
        print(f"{dir_} has {count} number of validation images")
    print(f"\nTotal validation images are {total_validation_images}\n")

    # Creating a function for using for show some images from each categories
    def imageshow(category_name, num_images=9):
        plt.figure(figsize=(8,8))
        category_p = os.path.join(train_path, category_name)
        if not os.path.exists(category_p):
            print(f"Category path {category_p} does not exist.")
            plt.close()
            return
        
        image_files = [img for img in os.listdir(category_p) if os.path.isfile(os.path.join(category_p, img))]
        image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # Filter for common image types

        display_count = min(len(image_files), num_images)
        if display_count == 0:
            print(f"No valid images found in {category_p}")
            plt.close() # Close the empty figure
            return

        for i_img in range(display_count):
            plt.subplot(3,3,i_img+1) # Assuming a 3x3 grid for 9 images
            img = load_img(os.path.join(category_p, image_files[i_img]), target_size=(48,48))
            plt.imshow(img)
            plt.axis('off')
        plt.suptitle(f"Sample Images: {category_name}",fontsize=20)
        sample_images_plot_p = os.path.join(base_output_path, f"sample_{category_name}_images.png")
        plt.savefig(sample_images_plot_p)
        print(f"Sample images plot saved to {sample_images_plot_p}")
        plt.show()
        plt.close()

    # Showing some images from a few categories (if they exist)
    if categories:
        # Show for up to 3 existing categories, or specific ones if known
        categories_to_show = [cat for cat in ['neutral', 'angry', 'surprise'] if cat in categories]
        if not categories_to_show and categories: # If specific ones not found, take first few
            categories_to_show = categories[:min(3, len(categories))]

        for cat_to_show in categories_to_show:
            imageshow(cat_to_show)
    else:
        print("No categories found in training data. Skipping image show.")
else:
    print("Error: Training or Validation path does not exist. Please check the paths.")
    print(f"Expected train path: {train_path}")
    print(f"Expected validation path: {val_path}")
    exit()

# --- Data Preprocessing ---
img_size = 48
batch_size = 64

datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    rescale=1./255
)

train_data = datagen_train.flow_from_directory(
    train_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    shuffle=True,
    color_mode='grayscale',
    class_mode='categorical'
)

datagen_validation = ImageDataGenerator(rescale=1./255)

validation_data = datagen_validation.flow_from_directory(
    val_path,
    target_size=(img_size,img_size),
    batch_size=batch_size,
    shuffle=False,
    color_mode='grayscale',
    class_mode='categorical'
)

num_classes = train_data.num_classes
print(f"Number of classes: {num_classes}")
print(f"Class indices: {train_data.class_indices}")


# --- Model Definition ---
model = Sequential()
# Block 1
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_1',input_shape=(img_size, img_size,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same', name = 'conv1_2'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), name = 'pool1_1'))
model.add(Dropout(0.3, name = 'drop1_1'))

# Block 2
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_1'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_2'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', name = 'conv2_3'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), name = 'pool2_1'))
model.add(Dropout(0.3, name = 'drop2_1'))

# Block 3
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_1'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_2'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_3'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv3_4'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), name = 'pool3_1'))
model.add(Dropout(0.3, name = 'drop3_1'))

# Block 4
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_1'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_2'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_3'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', name = 'conv4_4'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), name = 'pool4_1'))
model.add(Dropout(0.3, name = 'drop4_1'))

model.add(Flatten(name = 'flatten'))

model.add(Dense(256, name='fc1'))
model.add(BatchNormalization(name='bn_fc1'))
model.add(Activation('relu', name='relu_fc1'))
model.add(Dropout(0.25, name='drop_fc1'))

model.add(Dense(512, name='fc2'))
model.add(BatchNormalization(name='bn_fc2'))
model.add(Activation('relu', name='relu_fc2'))
model.add(Dropout(0.25, name='drop_fc2'))

model.add(Dense(num_classes, activation='softmax', name = 'output'))

opt = Adam(learning_rate=0.0007)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

try:
    utils.plot_model(model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
    print(f"Model architecture plot saved to {model_plot_path}")
except ImportError:
    print("Could not plot model architecture. graphviz and pydot might be required (pip install pydot graphviz).")
except Exception as e:
    print(f"An error occurred while plotting the model: {e}")


# --- Training ---
epochs = 100 # As per our code, can be reduced for faster testing. For actual training, 100 is reasonable.
# Consider reducing for a quick test run, e.g., epochs = 3
# epochs = 3 # For a quick test

steps_per_epoch = train_data.n // train_data.batch_size
validation_steps = validation_data.n // validation_data.batch_size

if train_data.n == 0:
    print("Error: No training images found. Please check the train_path and its subdirectories.")
    exit()
if validation_data.n == 0:
    print("Error: No validation images found. Please check the val_path and its subdirectories.")
    exit()


if steps_per_epoch == 0:
    print(f"Warning: steps_per_epoch is 0 (train_data.n={train_data.n}, batch_size={batch_size}). This means the training dataset size is smaller than the batch size.")
    print("Adjust batch_size or increase dataset size. Setting steps_per_epoch=1 for now.")
    steps_per_epoch = 1 # Ensure it's at least 1 to avoid errors, though training might be ineffective.
if validation_steps == 0:
    print(f"Warning: validation_steps is 0 (validation_data.n={validation_data.n}, batch_size={batch_size}). This means the validation dataset size is smaller than the batch size.")
    print("Adjust batch_size or increase dataset size. Setting validation_steps=1 for now.")
    validation_steps = 1 # Ensure it's at least 1.


print(f"Training steps per epoch: {steps_per_epoch}")
print(f"Validation steps per epoch: {validation_steps}")

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001, mode='auto', verbose=1) # Adjusted patience and min_lr
checkpoint = ModelCheckpoint(
    model_weights_path, # Path already updated to use ".weights.h5"
    monitor='val_accuracy',
    save_weights_only=True,
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001, # Small delta to ensure meaningful improvement
    patience=10,      # Increased patience
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint, reduce_lr, early_stopping]

print("\n--- Starting Model Training ---")
history = model.fit(
    x=train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data = validation_data,
    validation_steps = validation_steps,
    callbacks=callbacks,
    verbose=1
)
print("--- Model Training Finished ---")

# If restore_best_weights=True in EarlyStopping, the model already has the best weights.
# Otherwise, load them from the checkpoint if you want the absolute best based on val_accuracy.
# Since EarlyStopping restores based on val_loss, and Checkpoint saves based on val_accuracy,
# you might choose to explicitly load from checkpoint if val_accuracy is your primary metric for the saved model.
# However, `restore_best_weights=True` with `val_loss` is a common and robust approach.

model.save(final_model_save_path)
print(f"Trained model (potentially with best weights restored by EarlyStopping) saved to {final_model_save_path}")

# --- Post-Training Analysis ---

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(history_csv_path, index=False)
print(f"Training history saved to {history_csv_path}")
print("\nTraining History DataFrame head:")
print(history_df.head())

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(training_plots_path)
print(f"Training performance plots saved to {training_plots_path}")
plt.show()
plt.close()


# Predictions on validation set
print("\n--- Evaluating Model on Validation Data ---")
# The 'model' instance should have the best weights if early_stopping's restore_best_weights=True worked.
# Or, you can load the explicitly saved full model:
# model_for_eval = load_model(final_model_save_path)
# For this script, we'll continue using the 'model' instance.

# Ensure all validation data is predicted.
# Calculate steps needed to cover all samples.
val_predict_steps = (validation_data.n + validation_data.batch_size - 1) // validation_data.batch_size

predictions = model.predict(validation_data, steps=val_predict_steps, verbose=1)
# Trim predictions if predict generated more than validation_data.n samples due to batching
if len(predictions) > validation_data.n:
    predictions = predictions[:validation_data.n]

np.save(predictions_npy_path, predictions)
print(f"Validation predictions saved to {predictions_npy_path}")

y_pred_classes = np.argmax(predictions, axis=-1)
y_true_classes = validation_data.classes[:len(y_pred_classes)] # Ensure y_true aligns with potentially trimmed predictions

# Accuracy Score
final_accuracy = accuracy_score(y_true=y_true_classes, y_pred=y_pred_classes)
print(f"\nValidation Accuracy Score of the Trained Model: {final_accuracy:.4f}")

# Classification Report
class_names = list(validation_data.class_indices.keys())
report_dict= {}
try:
    report_str = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0, output_dict=True)
    print("\nClassification Report:")
    print(report_str)
except ValueError as e:
    print(f"\nError generating classification report: {e}")
    print("This can happen if some classes in y_true_classes have no corresponding predictions or true samples.")
    report_str = f"Could not generate full classification report due to: {e}"
    report_dict = {"error": str(e)}

accuracy_report_text_path = os.path.join(base_output_path, "accuracy_classification_report.txt") # Original text report
classification_report_json_path = os.path.join(base_output_path, "classification_report.json") # NEW JSON report path

# Save accuracy and classification report to a file
with open(accuracy_report_path, "w") as f:
    f.write(f"Validation Accuracy Score: {final_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report_str)
print(f"Accuracy and classification report (text) saved to {accuracy_report_path}")

# Save the report dictionary as JSON
with open(classification_report_json_path, "w") as f:
    json.dump(report_dict, f, indent=4)
print(f"Classification report (JSON) saved to {classification_report_json_path}")


# Confusion Matrix
def plot_confusion_matrix(cm, classes_list, fig_title='Confusion matrix', cmap_plt=plt.cm.Blues, normalize_cm=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize_cm=True`.
    """
    if normalize_cm:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Handle cases where sum is 0 (to avoid NaN)
        cm_normalized = np.nan_to_num(cm_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        print("Normalized confusion matrix")
        cm_to_plot = cm_normalized
    else:
        print('Confusion matrix, without normalization')
        cm_to_plot = cm

    plt.figure(figsize=(10,10))
    plt.imshow(cm_to_plot, interpolation='nearest', cmap=cmap_plt)
    plt.title(fig_title)
    plt.colorbar()
    tick_m = np.arange(len(classes_list))
    plt.xticks(tick_m, classes_list, rotation=45)
    plt.yticks(tick_m, classes_list)

    fmt = '.2f' if normalize_cm else 'd'
    thresh_val = cm_to_plot.max() / 2.
    for i_cm, j_cm in itertools.product(range(cm_to_plot.shape[0]), range(cm_to_plot.shape[1])):
        plt.text(j_cm, i_cm, format(cm_to_plot[i_cm, j_cm], fmt),
                horizontalalignment="center",
                color="white" if cm_to_plot[i_cm, j_cm] > thresh_val else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(num_classes)) # ensure labels cover all classes
np.set_printoptions(precision=2) # For printing the matrix if needed

# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes_list=class_names, fig_title='Normalized Confusion Matrix')
plt.savefig(confusion_matrix_plot_path)
print(f"Normalized confusion matrix plot saved to {confusion_matrix_plot_path}")
plt.show()
plt.close()

print("\n--- Script Finished ---")