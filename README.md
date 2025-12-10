# Handwritten Digit Recognizer (MNIST)

This repository contains a Jupyter Notebook (p1_Handwritten_Digit_Recognizer.ipynb) that trains and evaluates two models to recognize handwritten digits from the MNIST dataset:

- A fully-connected Multi-Layer Perceptron (MLP)
- A Convolutional Neural Network (CNN)

The notebook also includes a small Streamlit demo script to load the trained CNN and run predictions on uploaded images.

Notebook (permalink):
https://github.com/ishhverma/artificial-intelligence-projects/blob/df07bfbf3509da1c0659fdf6a7cf6adbf3065fbb/p1_Handwritten_Digit_Recognizer.ipynb

## Contents

- p1_Handwritten_Digit_Recognizer.ipynb — the notebook that:
  - Loads and visualizes the MNIST dataset
  - Normalizes and reshapes data
  - Trains an MLP on flattened images
  - Trains a CNN on 28×28 grayscale images
  - Compares test performance (loss & accuracy)
  - Saves trained models (`mlp_model.keras` and `cnn_model.keras`)
  - Contains a Streamlit app example for inference

## Results (from the notebook)

- MLP Test Loss: 0.0697  
- MLP Test Accuracy: 0.9794

- CNN Test Loss: 0.0259  
- CNN Test Accuracy: 0.9927

(Your exact results may vary depending on environment, package versions, and randomness.)

## Requirements

The notebook was developed with standard Python ML packages. Minimum / typical packages:

- Python 3.8+
- tensorflow (2.x)
- numpy
- matplotlib
- pillow (PIL)
- streamlit (only if you want to run the demo app)

Install example (recommended in a venv):

pip install tensorflow numpy matplotlib pillow streamlit

Or, for CPU-only TensorFlow:

pip install "tensorflow-cpu" numpy matplotlib pillow streamlit

## How to run

1. Open and run the notebook
   - Option A — Google Colab: Click "Open in Colab" from the notebook header (link included in the notebook).
   - Option B — Locally: Clone the repository, install dependencies, then open the notebook with JupyterLab/Notebook and run the cells in order.

2. Train models
   - The notebook trains both the MLP and CNN (each for 10 epochs in the example). Training outputs and final test metrics are printed in the notebook.

3. Saved models
   - After training, the notebook saves:
     - `mlp_model.keras`
     - `cnn_model.keras`
   - Keep these files in the same directory as the Streamlit app if you want to run it locally.

4. Run the Streamlit demo (optional)
   - Save the Streamlit code from the notebook into a file called `app.py`.
   - Ensure `cnn_model.keras` is present in the same directory.
   - Run:
     streamlit run app.py
   - Open the URL shown by Streamlit in your browser and upload an image of a handwritten digit (jpg/png) to get a prediction and confidence.

Notes about the demo:
- The notebook includes a simple preprocessing routine: resize to 28×28, convert to grayscale, normalize to [0,1], and reshape to (1, 28, 28, 1).
- The demo expects a single digit centered in the image for best results.

## Tips & Caveats

- Reproducibility: For exact reproducibility, set random seeds and fix package versions.
- Running on GPU will speed up training. If you run on CPU, training will take longer.
- The notebook uses a validation split (0.2) during training — adjust if you prefer different splits or use a dedicated validation set.
- The notebook saves the entire model (including optimizer state). When loading only for inference, possible warnings about optimizer variable loading may appear; these can usually be ignored.

## Suggested improvements

- Add data augmentation (for robustness to rotations/translation).
- Add command-line script to run inference on a folder of images.
- Export the model to a lighter format (e.g., TensorFlow Lite) for mobile/edge deployment.
- Add unit tests or small example images for the demo.

## License & Contributions

- License: add your preferred license (e.g., MIT) to the repository if you want to make it open-source.
- Contributions: feel free to open issues or PRs in this repository to suggest improvements.

---

I created a README.md that summarizes the notebook, lists requirements and run instructions, and highlights results and next steps. If you want, I can:
- generate a requirements.txt,
- extract the Streamlit script into a separate app.py file and add it to the repo,
- or produce a short CONTRIBUTING.md template. Which should I do next?
