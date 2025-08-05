# 🍽️ Dish Detect

**Dish Detect** is a lightweight machine learning project that predicts the type of dish—**Pizza**, **Sushi**, or **Shawarma**—based on a few descriptive questions. It leverages classical NLP and classification models like **Logistic Regression** and **Naive Bayes**, enhanced with techniques such as **TF-IDF vectorization** and **K-Fold Cross Validation**.

---

## 🧠 Features

- Predicts one of three food categories: Pizza, Sushi, or Shawarma.
- Uses logistic regression and multinomial naive Bayes classifiers.
- Performs model evaluation with K-Fold Cross-Validation.
- Built with `scikit-learn`, `pandas`, and `numpy`.
- Includes pre-trained model assets for fast inference.

---

## 📦 Dependencies

The project uses the following Python libraries:

```python
pandas
numpy
scikit-learn
json
pathlib
```
---

## 🚀 Usage

1. **Install dependencies**  
   Make sure you have Python ≥ 3.7 and the required libraries:

   ```bash
   pip install pandas numpy scikit-learn
   ```
2. **Train the model**
   
   You can run the training script:
   
   ```
   python prediction_script.py
   ```
   This will train logistic regression and naive Bayes models on your dataset.
   
3. **Make predictions**

    Use pred.py to load trained models from model_assets and run inference based on question inputs.
    ```
    python pred.py
    ```

---

## 💡 Notes
- The model_assets folder (zipped) contains the trained models.
- You can customize the question set or expand the classification categories with more training data.
- The model is purely based on text inputs and is ideal for fast prototyping or educational ML demos.
---

## ✨ Author
Created by Aastha Sharma — April 2025.
