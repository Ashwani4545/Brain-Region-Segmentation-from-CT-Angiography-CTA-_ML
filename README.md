🧠 Brain Region Segmentation from CT Angiography (CTA) Images

🎯 Project Overview
This project aims to design an Artificial Intelligence-based system for accurate brain region segmentation from CT Angiography (CTA) images. It helps in identifying critical regions (arteries, veins, ischemic regions) in the brain for medical analysis and diagnosis.

🚀 Features
Preprocessing of CTA images (resampling, normalization, cropping).

Data augmentation to improve model robustness.

Deep learning-based segmentation using 3D U-Net architecture.

Evaluation with key metrics: Dice Coefficient, Jaccard Index, Hausdorff Distance.

Post-processing to refine segmentation masks.

Simple inference pipeline to segment new CTA images.

📚 Technologies and Libraries Used
Python 3.x

PyTorch / TensorFlow

MONAI (Medical Open Network for AI)

NumPy, Pandas

OpenCV, SimpleITK

pydicom / nibabel (DICOM/NIfTI support)

Matplotlib (visualization)

Scikit-learn (metrics)

📁 Dataset
Source:

The Cancer Imaging Archive (TCIA)

MICCAI Challenge datasets

Publicly available CTA datasets in DICOM/NIfTI format.

Data Format:

Input: 3D DICOM / NIfTI CTA image volumes.

Output: Corresponding segmentation masks (binary or multi-class).

🧱 System Architecture
[Input CTA Image (DICOM/NIfTI)]
↓
[Preprocessing (Resample, Normalize, Crop)]
↓
[Data Augmentation (Flipping, Rotation, Scaling)]
↓
[3D U-Net Deep Learning Model]
↓
[Segmentation Mask Output]
↓
[Postprocessing (Morphological Clean-up)]
↓
[Final Segmentation Visualization]

✅ Usage Instructions
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Preprocess Data
python preprocess.py --input_dir ./data/raw --output_dir ./data/processed

3️⃣ Train the Model
python train.py --data_dir ./data/processed --epochs 100 --batch_size 2

4️⃣ Evaluate the Model
python evaluate.py --model ./saved_models/best_model.pth --data_dir ./data/processed

5️⃣ Run Inference on New Images
python inference.py --model ./saved_models/best_model.pth --input ./new_cta_image.nii --output ./segmented_mask.nii

📊 Evaluation Metrics
Dice Coefficient (DSC)
Jaccard Index (IoU)
Hausdorff Distance
Sensitivity & Specificity

⚡ Future Improvements
Multi-modal data input (CT + MRI).
Semi-supervised learning to use unlabeled data.
Web-based GUI for easier usage by medical professionals.

📖 References
U-Net Paper
MONAI Framework
Medical Image Datasets

📝 License
MIT License

👤 Author
Ashwani Pandey
