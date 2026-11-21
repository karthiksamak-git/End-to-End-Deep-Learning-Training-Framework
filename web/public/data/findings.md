# Error analysis

Task: `image` · samples: 2000 · misclassified: 243

- **2000** held-out samples, **243** misclassified (12.15% error rate).
- Overall accuracy **0.8785**, macro F1 **0.8757**, ROC-AUC (macro OVR) **0.9883**.
- Weakest class by recall: **Shirt** (recall 0.520, support 200).
- Most frequent confusion: **Shirt -> T-shirt/top** (48 cases).
- Expected calibration error **0.0143** over 10 confidence bins (mean confidence 0.885 vs accuracy 0.878).
- Most confident error: true **Sneaker** predicted **Ankle boot** with confidence 0.998.
