import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Dataset path
# -------------------------------

DATASET_PATH = "train"

# -------------------------------
# 2. Parameters
# -------------------------------

IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 1000

images = []
labels = []

# -------------------------------
# 3. Load images
# -------------------------------

cat_count = 0
dog_count = 0

for filename in os.listdir(DATASET_PATH):

    if filename.endswith(".jpg"):

        # Identify class from filename
        if filename.startswith("cat"):

            if cat_count >= MAX_IMAGES_PER_CLASS:
                continue

            label = 0
            cat_count += 1

        elif filename.startswith("dog"):

            if dog_count >= MAX_IMAGES_PER_CLASS:
                continue

            label = 1
            dog_count += 1

        else:
            continue

        # Read image
        image_path = os.path.join(DATASET_PATH, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize pixel values
        image = image / 255.0

        # Flatten image into 1D array
        image = image.flatten()

        images.append(image)
        labels.append(label)


# Convert to NumPy arrays
X = np.array(images)
y = np.array(labels)

print("Total images:", len(X))
print("Feature size:", X.shape)
print("Cats:", np.sum(y == 0))
print("Dogs:", np.sum(y == 1))


# -------------------------------
# 4. Split dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


# -------------------------------
# 5. Create SVM model
# -------------------------------

model = SVC(
    kernel="linear",
    C=1
)


# -------------------------------
# 6. Train SVM
# -------------------------------

print("Training SVM...")

model.fit(X_train, y_train)

print("Training completed!")


# -------------------------------
# 7. Prediction
# -------------------------------

y_pred = model.predict(X_test)


# -------------------------------
# 8. Accuracy
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# -------------------------------
# 9. Classification report
# -------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# -------------------------------
# 10. Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# -------------------------------
# 11. Display confusion matrix
# -------------------------------

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j],
                 ha="center",
                 va="center")

plt.show()