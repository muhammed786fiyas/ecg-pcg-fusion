## **General Processing Pipeline**

The following pipeline is used to ensure **patient-wise separation**, **no data leakage**, and **fair model evaluation**.

**Step 1: Patient-wise split**
- Split the dataset into **70% training** and **30% testing** patients at the record level.

**Step 2: Training data processing**
- Segment ECG–PCG signals into **3-second windows**.
- Apply **data augmentation** to increase training diversity.
- Generate **scalogram representations** from the augmented segments.

**Step 3: Testing data processing**
- Segment ECG–PCG signals into **3-second windows**.
- Generate **scalogram representations**.
- **No data augmentation** is applied to test data.

**Step 4: Model training**
- Train deep learning models using **training scalograms only**.

**Step 5: Model evaluation**
- Evaluate trained models on **test scalograms** corresponding to unseen patients.
