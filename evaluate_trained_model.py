import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd # For potential future use, not strictly needed for just CM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os
import itertools

# --- Configuration ---
# Path to the validation data
base_data_path = r"C:\Users\HP\Desktop\FER2013"
val_path = os.path.join(base_data_path, "valid")

# Path where the trained model is saved
base_output_path = r"C:\Users\HP\Desktop\desk"
final_model_save_path = os.path.join(base_output_path, "CNN_Model_emotion_trained.h5")

# Output paths for this evaluation script
eval_accuracy_report_path = os.path.join(base_output_path, "evaluation_accuracy_report.txt")
eval_confusion_matrix_plot_path = os.path.join(base_output_path, "evaluation_normalized_confusion_matrix.png")

# Model and data parameters (should match training)
img_size = 48
batch_size = 64 # Use the same batch size as training for validation_data generator, or larger if memory allows for prediction

# --- Function to Plot Confusion Matrix (copied from your training script) ---
def plot_confusion_matrix(cm, classes_list, fig_title='Confusion matrix', cmap_plt=plt.cm.Blues, normalize_cm=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize_cm=True`.
    """
    if normalize_cm:
        cm_float = cm.astype('float')
        cm_sum = cm_float.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm_float, cm_sum, out=np.zeros_like(cm_float), where=cm_sum!=0)
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
    plt.xticks(tick_m, classes_list, rotation=45, ha="right")
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

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    print("--- Starting Evaluation of Trained Model ---")

    # 1. Check if model file exists
    if not os.path.exists(final_model_save_path):
        print(f"ERROR: Trained model not found at {final_model_save_path}")
        print("Please ensure the model has been trained and saved correctly.")
        exit()

    # 2. Load the trained model
    print(f"Loading model from: {final_model_save_path}")
    try:
        model = load_model(final_model_save_path)
        print("Model loaded successfully.")
        model.summary() # Optional: display model summary
    except Exception as e:
        print(f"ERROR: Could not load model. Error: {e}")
        exit()

    # 3. Prepare Validation Data
    if not os.path.exists(val_path):
        print(f"ERROR: Validation data path not found: {val_path}")
        exit()

    datagen_validation = ImageDataGenerator(rescale=1./255)
    validation_data = datagen_validation.flow_from_directory(
        val_path,
        target_size=(img_size,img_size),
        batch_size=batch_size, # Can be larger for prediction if memory allows
        shuffle=False,        # IMPORTANT: Shuffle must be False for correct evaluation
        color_mode='grayscale',
        class_mode='categorical' # Still needed for .classes and .class_indices
    )

    if validation_data.n == 0:
        print("ERROR: No images found in the validation directory.")
        exit()

    num_classes = validation_data.num_classes
    class_names_map = validation_data.class_indices
    # Create class_names list in the correct order
    class_names = [""] * num_classes
    for name, index in class_names_map.items():
        class_names[index] = name

    print(f"Found {validation_data.n} validation images belonging to {num_classes} classes.")
    print(f"Class names: {class_names}")


    # 4. Make Predictions on Validation Data
    print("\nMaking predictions on validation data...")
    # Calculate steps needed to cover all samples.
    val_predict_steps = (validation_data.n + validation_data.batch_size - 1) // validation_data.batch_size
    try:
        predictions = model.predict(validation_data, steps=val_predict_steps, verbose=1)
    except Exception as e:
        print(f"ERROR: Failed to make predictions. Error: {e}")
        exit()

    # Trim predictions if predict generated more than validation_data.n samples
    if len(predictions) > validation_data.n:
        predictions = predictions[:validation_data.n]

    y_pred_classes = np.argmax(predictions, axis=-1)
    y_true_classes = validation_data.classes[:len(y_pred_classes)] # Ensure y_true aligns


    # 5. Calculate and Display/Save Metrics
    if len(y_true_classes) > 0 and len(y_pred_classes) > 0:
        # Accuracy Score
        final_accuracy = accuracy_score(y_true=y_true_classes, y_pred=y_pred_classes)
        print(f"\nValidation Accuracy Score: {final_accuracy:.4f}")

        # Classification Report
        report_str = "Classification report not generated."
        try:
            report_str = classification_report(y_true_classes, y_pred_classes, target_names=class_names, zero_division=0)
            print("\nClassification Report:")
            print(report_str)
        except Exception as e:
            print(f"\nError generating classification report: {e}")
            report_str = f"Could not generate full classification report due to: {e}"

        # Save accuracy and classification report to a file
        try:
            with open(eval_accuracy_report_path, "w") as f:
                f.write(f"Evaluation of model: {final_model_save_path}\n")
                f.write(f"Validation Accuracy Score: {final_accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report_str)
            print(f"Evaluation report saved to {eval_accuracy_report_path}")
        except Exception as e:
            print(f"ERROR: FAILED to write evaluation report. Error: {e}")

        # Compute and Plot Confusion Matrix
        print("\nGenerating Confusion Matrix...")
        cnf_matrix = confusion_matrix(y_true_classes, y_pred_classes, labels=np.arange(num_classes))
        np.set_printoptions(precision=2)

        plot_confusion_matrix(cnf_matrix, classes_list=class_names, fig_title='Evaluation - Normalized Confusion Matrix')
        try:
            plt.savefig(eval_confusion_matrix_plot_path)
            print(f"Confusion matrix plot saved to {eval_confusion_matrix_plot_path}")
        except Exception as e:
            print(f"ERROR: Could not save confusion matrix plot. Error: {e}")
        plt.show() # Display the plot
        plt.close()
    else:
        print("Warning: y_true_classes or y_pred_classes are empty. Cannot calculate metrics.")

    print("\n--- Evaluation Script Finished ---")