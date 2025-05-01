# Disease Incidence Analysis Using Hadoop Streaming (Simulated)

## Domain: Healthcare

### Tools Used:
- **Python**
- **Simulated MapReduce (Hadoop Streaming)**

### Project Overview:
This project simulates how a **Hadoop-based MapReduce job** can be used to analyze **healthcare data**, specifically disease incidence records. The dataset used in this project consists of disease records with various health attributes. We use Python-based **mapper** and **reducer** scripts to process this data, mimicking the Hadoop streaming logic.

While Hadoop’s full functionality isn't implemented in this project due to environment limitations, the **MapReduce** simulation allows for scalable data processing in a distributed system, making it possible to run the computations on large datasets.

### Team Members:
- **Aakashsingh Rajput** 
- **Md. Faraaz Ahmed** 

---

### Dataset:
The dataset (`disease_data.csv`) contains records of different diseases, along with symptoms and health indicators. A sample record looks like this:

| Disease          | Fever | Cough | Fatigue | Difficulty Breathing | Age | Gender | Blood Pressure | Cholesterol Level | Outcome Variable |
|------------------|-------|-------|---------|----------------------|-----|--------|-----------------|--------------------|------------------|
| Influenza        | Yes   | No    | Yes     | Yes                  | 19  | Female | Low             | Normal             | Positive         |
| Common Cold      | No    | Yes   | Yes     | No                   | 25  | Female | Normal          | Normal             | Negative         |
| Asthma           | Yes   | Yes   | No      | Yes                  | 25  | Male   | Normal          | Normal             | Positive         |
| Eczema           | Yes   | No    | No      | No                   | 25  | Female | Normal          | Normal             | Positive         |

---

### Simulated MapReduce Process:

The workflow of this project is broken down into **two scripts** that simulate the mapper-reducer pattern commonly used in Hadoop's MapReduce framework:

1. **mapper.py**: Reads the input dataset line-by-line, processes the records, and outputs key-value pairs (`disease_name -> 1`). The mapper simulates the task of counting occurrences of each disease.
   
2. **reducer.py**: Aggregates the counts from the mapper, summing the occurrences of each disease, and prints out the final disease frequency counts.

3. **run_simulation.py**: Orchestrates the simulation by invoking the `mapper.py` and `reducer.py` scripts using subprocess calls, passing the data between them, and capturing the output.

---

### Key Outputs:
- **Disease Frequency**: The main output of the MapReduce process is the frequency count of diseases, which shows the number of times each disease occurs in the dataset.
  
---

### Graphs and Analysis:

The following graphs are generated using the disease frequency data to provide insights into the most common health issues.

#### 1. **Disease Frequency Distribution**:
- **Graph Type**: Bar Plot
- **Description**: This graph shows the frequency count of diseases, including the top 5 and least 5 diseases from the dataset.
- **Explanation**: We first calculate the frequency of each disease in the dataset, then display the top 5 and least 5 diseases with the highest and lowest occurrence counts. This helps in identifying which diseases are more prevalent and which are less common.

---

#### 2. **Disease Incidence Heatmap (Confusion Matrix)**:
- **Graph Type**: Heatmap (Confusion Matrix)
- **Description**: This heatmap visualizes the classification performance of a machine learning model in predicting the outcome variable (disease diagnosis) based on the health attributes.
- **Explanation**: The confusion matrix helps to evaluate the accuracy of the machine learning model, displaying how often the model's predictions align with the actual disease outcomes.

---

#### 3. **ROC Curve**:
- **Graph Type**: ROC Curve
- **Description**: The ROC curve displays the trade-off between the **True Positive Rate** (sensitivity) and **False Positive Rate** (1-specificity) at various threshold settings.
- **Explanation**: The curve helps in evaluating how well the model distinguishes between positive and negative outcomes for the disease diagnosis task.

---

#### 4. **Precision-Recall Curve**:
- **Graph Type**: Precision-Recall Curve
- **Description**: This graph plots **Precision** (positive predictive value) against **Recall** (sensitivity) for different thresholds.
- **Explanation**: It’s especially useful in imbalanced datasets to evaluate the model’s performance concerning the true positive and false negative rates.

---

#### 5. **Feature Importance Bar Chart**:
- **Graph Type**: Feature Importance (Bar Plot)
- **Description**: This bar plot shows the importance of each feature in predicting the disease outcome, based on the trained machine learning model (Random Forest).
- **Explanation**: The feature importance tells us which health indicators (e.g., age, fever, fatigue, etc.) have the most significant impact on disease prediction.

---

### Conclusion:
- The **Disease Frequency Distribution** graph highlights the most and least common diseases in the dataset.
- The **Machine Learning Model** evaluates disease prediction accuracy, and the associated graphs (ROC Curve, Precision-Recall Curve, etc.) show the model's performance.
- The **Feature Importance Chart** reveals which attributes most influence the disease prediction.

By running the above simulations, we gain valuable insights into disease prevalence and predictive modeling in the healthcare domain.

---

### How to Run the Project:
1. Clone or download the project files.
2. Install necessary Python libraries:
   - `pandas`
   - `sklearn`
   - `matplotlib`
   - `seaborn`
3. Run the **`run_simulation.py`** script to start the MapReduce simulation and disease frequency analysis.
4. View the graphs generated in the output.

---

