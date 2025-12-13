# Assessing_Fruit_Quality_with_Computer_Vision

**1. Environment Setup**

This project is designed to run on Google Colab to leverage the T4 GPU and High-RAM environment.

  * Runtime: Python 3.x with GPU acceleration (T4).
  * Key Dependencies:
    * tensorflow
    * matplotlib (for plotting)
    * scikit-learn (for visualization)
    * numpy
    * os
    * keras-tuner (for hyperparameter optimization)
  * Mixed Precision: The global policy is set to mixed_float16 to optimize memory and speed.
***

**2. Step-by-Step Execution**

To reproduce the results, execute the notebooks in the following order:
* Data Preparation
  * Mount your Google Drive and ensure your dataset is structured into _train_, _test_, and _validate_ folders, with each class in its own subfolder within parent folder.
  * Update the _base_dir_ variable in the notebooks to point to your Drive path.
* Run Baseline Model (_Baseline Model.ipynb_):
  * Trains a ResNet50 from scratch.
  * Expect high training accuracy but significant overfitting.
* Run Transfer Learning (_Transfer-Learning.ipynb_):
  * Uses EfficientNetB0 with pretrained imagenet weights
  * Freezes all model layers, then unfreezes last 30% of layers
* Run Hyperparameter Tuning (_Hyperparameter tuning.ipynb_):
  * Optimizes to find best learning rate, dropout rate, and freezed layers
* Run Final Model (_Final Model.ipynb_):
  * Loads the pre-trained EfficientNetB0 backbone.
  * Applies the tuned hyperparameters.

***
**3. Expected Outputs**

* You should see the progress bar for 10 epochs. The final model should report a Test Accuracy of ~91%.
* A file named _final__model_with_hp.keras_ will be generated in your working directory.
* The notebook will display a classification report, confusion matrix, and a 15-image grid of misclassified samples, showing the True vs. Predicted labels for qualitative error analysis.

***
**4. Reproducing Results**

To achieve the exact 0.91 F1-Score shown in our report:
* Ensure the BATCH_SIZE is set to 32 for the final model.
* Ensure the target_size is exactly (224, 224).
* Verify that the last 80 layers are trainable, dropout is 0.1, and learning rate is 0.00018791
