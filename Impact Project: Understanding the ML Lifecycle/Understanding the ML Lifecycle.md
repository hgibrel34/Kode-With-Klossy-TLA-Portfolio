I'm an Industrial and Systems Engineering student at Virginia Tech, currently participating in the Break Through Tech AI/ML fellowship, where I'm working toward a Machine Learning certificate. Since my coding skills had gone dormant without regular use in my coursework, jumping straight into this program was an adjustment. The course is structured around the ML lifecycle, from business understanding through exploratory data analysis to deployment; taking in all of these steps at once, especially the core concepts, libraries, and functions behind each stage, was a lot. A guide that connected the big picture of the lifecycle to my own progress through it would have made it much easier for me and my peers.

This guide walks through each stage of the ML lifecycle, connecting the core concept to the code you'd actually write to apply it.
![[Impact Project - Understanding the ML Lifecycle/Images/cycle diagram.png|366]] 
*Source: [michael-fuchs-python](https://michael-fuchs-python.netlify.app/2020/08/21/the-data-science-process-crisp-dm/)*

## Table of Contents

- [Framing: Supervised vs Unsupervised Learning](#framing-supervised-vs-unsupervised-learning)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling & Optimization](#modeling--optimization)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Iteration](#iteration)

## Framing: Supervised vs Unsupervised Learning
---
**Core concept:** Before touching the data, you must first ask yourself: what kind of problem am I actually solving? This shapes all your choices down the line: the data that you need, the libraries you'll use, and the kind of model you'll build.
![[Impact Project - Understanding the ML Lifecycle/Images/framing image.png]]

**Vocabulary:**

- **Examples:** The rows of your dataset, individual data points.
- **Features:** The columns, the variables the model uses to make predictions.
- **Labels:** The target variable you're trying to predict.

#### Supervised Learning

Supervised learning uses labeled examples: data where each row has features and a known answer. The model's objective is to learn the relationship between features and labels of past data, so it can predict the label for unseen data. The model compares its predictions against the true label to improve over time.

**Characteristics:**

- Each input has a known output.
- The model evaluates its own performance to improve predictions.
- The model aims to make accurate predictions on new data.

**Types of Supervised Learning:**

- **Classification:** the model predicts a categorical outcome, sorting data into discrete classes. _Example: is this transaction fraudulent or not?_
- **Regression:** the model predicts a continuous numerical value. _Example: what will this house sell for?_

#### Unsupervised Learning

Unsupervised Learning uses unlabeled examples, no pre-defined labels or categories. Instead of learning to predict a known answer, the algorithm discovers patterns on its own, usually through grouping similar examples together.

**Characteristics:**

- No known output to compare against.
- The algorithm looks for structure in the data itself.
- Success is measured by how useful the discovered groupings are.

**Types of Unsupervised Learning:**

- **Clustering:** the model groups examples together based on how similar their features are, without being told ahead of time what the groups represent. _Example: grouping customers by purchasing behavior to reveal segments the business didn't predefine._

**Think about:**

- Is this actually a machine learning problem, and if so, is it supervised or unsupervised? What would tell you whether you have labels to work with?
- How would misclassifying the problem type early affect every stage downstream?
- Can you think of a problem that could be framed either way, depending on what data you have access to?

## Business Understanding
---
**Core Concepts:** In order to define a target variable or pick a model, you need to first translate a business goal into a specific, solvable ML problem. A model doesn't exist on its own, it's built to support an organization's goals. At this point, you figure out what the goal is, and if ML is the right tool.

**This stage centers on four questions you must answer before moving forward:**

- **What is the organization's objective?** — the underlying business goal
- **What outcome (label) should be produced to help accomplish the objective?** — sometimes predefined, sometimes not
- **What data (features) can help generate the prediction?** — information that plausibly relates to the outcome
- **Is Machine Learning the right approach?** — ML works when patterns are too complex, high-volume, or inconsistent to define by hand

**Let's look at an example where ML works:**

A food delivery app wants to reduce late deliveries **(objective)**. To do that, they might predict whether a given order will arrive late **(outcome/label)**, using features like distance, time of day, restaurant prep time, and driver availability **(features)**.

A simple rule, "flag any order over 5 miles," might catch some cases, but it ignores how prep time, traffic, and driver load interact, which is exactly the type of complex, data-rich pattern ML is suited for **(why ML)**.


**Think about:**

- Is this actually a machine learning problem? When would a simple rule or lookup table solve it just as well?
- What does success look like in business terms, and does it match the model metric you'll evaluate against later?
- Who are the stakeholders, and what constraints (interpretability, regulation, etc.) shape which models are even on the table?

## Data Understanding
---
**Core Concepts:** You need to understand the data you're working with before building. This stage is about exploring your data honestly, finding out where it falls short, and what patterns already exist. Skipping this step results in a model built on assumptions.

**What you're looking for:**

- **Outliers** — values that sit far outside the normal range. These can be errors, or they can be rare events worth investigating rather than removing.
- **Missing Data** — gaps in your dataset. Ignoring these gaps can bias your model if the gaps aren't random.
- **Class Imbalance** — when one category dominates your data significantly more than others. This results in a model that looks accurate overall but performs poorly on the underrepresented group.
- **Summary Statistics** — mean, median, standard deviation, and percentiles give you a big picture read on where your data sits and how spread out it is, before looking at a single row in detail.

| Concept            | Library / Function                                                | What It Does                                                         |
| ------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------- |
| Outliers           | `seaborn.boxplot()`, `seaborn.histplot()`, `matplotlib.scatter()` | Visualize distributions to find values outside the normal range      |
| Missing Data       | `.isnull().sum()` (pandas)                                        | Counts missing values per column                                     |
| Class Imbalance    | `.value_counts()` (pandas)                                        | Shows the distribution of a categorical or label column              |
| Summary Statistics | `.describe()` (pandas)                                            | Generates mean, median, std, and percentiles for all numeric columns |

**Notes:**

- This stage is diagnostic, not corrective. Your goal is to identify problems within your data, not solve them quite yet.
- By the end of this step, you'll have analyzed individual columns and relationships between columns.
- A good EDA makes every later stage easier.

**Example:** Imagine a dataset of loan applicants used to predict default risk. Before training anything, you'd check: are there applicants with impossible ages **(outliers)**? Are income values missing for a chunk of rows **(missing data)**? Are 95% of the labeled examples "did not default" **(class imbalance)**? Catching these now means you're not only confident in your model, but not surprised by a broken model later.

**Think about:**

- What would you do if you found a class imbalance or missing data pattern that wasn't random?
- How would you decide whether an outlier is an error to investigate or a rare event worth keeping?
- What questions about your data would you want answered before writing a single line of modeling code?

## Data Preparation
---
**Core Concept:**  This stage takes everything EDA reveals and fixes it, then splits your cleaned dataset so you can properly train and evaluate a model on separate data.

**Acting on what you found:**

- **Outliers** — either remove them, or cap them at a reasonable percentile instead of deleting (winsorizing), depending on whether they're errors or meaningful extremes.
- **Missing data** — drop the affected rows/columns, or fill them with a value (like the mean) depending on how much is missing and why.
- **Categorical features** — convert them into a numeric format a model can actually use (one-hot encoding is the most common).

**Splitting your data:**

Once your data is clean, you can't train and test on the same rows, that only tells you if the model memorized the data, not whether it actually learned something generalizable. Splitting your data creates a training set (what the model learns from) and a test set (unseen data used to check if it generalizes).

**Notes:**

- This stage directly determines how trustworthy your model's later evaluation will be.
- Decisions made here (how you handle outliers, how you split data) can't easily be undone once modeling starts.
- The right choice (drop vs. fill, remove vs. winsorize) typically depends on why the data is messy, not a fixed rule.

|Task|Library/Function|What It does|
|---|---|---|
|Remove or cap outliers|`.clip()` (pandas)|Caps values at a chosen percentile (winsorizing) instead of deleting them|
|Drop missing data|`.dropna()` (pandas)|Removes rows or columns with missing values|
|Fill missing data|`fillna()` (pandas)|Fills missing values with a specified value (e.g., mean, median)|
|Encode categorical features|`pd.get_dummies()` (pandas) or `OneHotEncoder()` (scikit-learn)|Converts categorical variables into a numeric (one-hot encoded) format|
|Split data|`train_test_split()` (scikit-learn)|Splits a dataset into training and test sets|

**Example:** Continuing the loan default example, you'd winsorize the impossible ages instead of deleting those applicants outright (in case the rest of their data is still useful), fill missing income values with a reasonable estimate, one-hot encode loan purpose (e.g., "auto," "home," "personal"), and then split the cleaned dataset 70/30 into training and test sets before modeling.

**Think about:**

- How would you decide between dropping missing data and filling it in? Are you considering 'why' the data is missing?
- Could your outlier handling accidentally introduce bias into your model?
- Why can't you undo the training/test split decision once modeling has started?

## Modeling & Optimization
---
**Core Concept:** This is the stage where your cleaned data gets used to build something. Modeling means choosing an algorithm suited to your problem, training it on your data, and using a loss function to measure and improve model performance. Optimization is the process of guiding the model toward better performance through the feedback loop.

**Choosing a model — a few common options and when they fit:**

| Model                     | Good for                                                           | Key idea                                                                                                                                                                     | Visual                                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KNN (K-Nearest Neighbors) | Classification, when similar examples likely share a label         | Makes predictions by looking at the K closest data points and going with the majority label. There is no real "training," just storing data and comparing at prediction time | ![[Impact Project - Understanding the ML Lifecycle/Images/KNN image.png]]*Source: [ResearchGate](https://www.researchgate.net/figure/sualization-of-k-Nearest-Neighbors-with-two-classes-blue-circles-and-red-triangles-with_fig3_359786522)* |
| Decision Trees            | Classification or regression, when you want an interpretable model | Splits data into branches based on feature values, groups similar examples together at each split                                                                            | ![[Impact Project - Understanding the ML Lifecycle/Images/DT image.png\|183]]*Source: [codementor](https://www.codementor.io/@mgalarny/visualizing-decision-trees-with-python-scikit-learn-graphviz-matplotlib-154mszcto7)*                   |
| Logistic Regression       | Binary classification                                              | Outputs a probability (0 to 1) that an example belongs to a class, based on a weighted combination of features                                                               | ![[Impact Project - Understanding the ML Lifecycle/Images/LG image.png]] *Source: [Medium](https://medium.com/analytics-vidhya/logistic-regression-b30ca0bec653)*                                                                             |
| Linear Regression         | Predicting a continuous number                                     | Fits a straight line (or hyperplane) through the data to model the relationship between features and a numeric label                                                         | ![[Impact Project - Understanding the ML Lifecycle/Images/LR image.png]]*Source: [python-graph-gallery](https://python-graph-gallery.com/556-visualize-linear-regression/)*                                                                   |


**Loss functions — how a model knows it's wrong:**

A loss function measures how far off a model's predictions are from the actual labels. High loss means bad predictions; low loss means the model is doing well. Different problem types use different loss functions:

- **Log loss** — used for classification, penalizes confident-but-wrong predictions 
- **Mean squared error (MSE)** — used for regression, squares the difference between predicted and actual values, bigger errors are penalized more

**Gradient descent- how a model actually improves:**

Training is iterative. Gradient descent starts with random model parameters (weights), checking the loss, then adjusting the parameters in whichever direction reduces the loss the most, repeating this until the loss stops meaningfully improving. 

Learning rate controls how big each step is. Too high, and you can overshoot the best point and never settle. Too low, and training takes forever to converge.

**Hyperparameters- the settings you choose before training:**

Unlike weights (which the model learns on its own), hyperparameters are values you set beforehand: how many neighbors K to use in KNN, how deep a decision tree can grow, what learning rate to use in gradient descent. Tuning these is part of your job, there's no universal "right" value, only what performs best for your specific data.

**Overfitting vs. underfitting — the balancing act:**

- **Overfitting** — the model is too complex, and learns quirks specific to the training data that don't generalize. Low training error, high test error.
- **Underfitting** — the model is too simple to capture real patterns. High error on both training and test data.
![[Impact Project - Understanding the ML Lifecycle/Images/|372]]
*Source: [SuperAnnotate](https://www.superannotate.com/blog/overfitting-and-underfitting-in-machine-learning)*

| Task                  | Library / Function                                                    | What It Does                                                            |
| --------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| K-Nearest Neighbors   | `KNeighborsClassifier()` (scikit-learn)                               | Predicts based on the majority label among the K closest data points    |
| Decision Trees        | `DecisionTreeClassifier()` / `DecisionTreeRegressor()` (scikit-learn) | Splits data into branches based on feature values                       |
| Logistic Regression   | `LogisticRegression()` (scikit-learn)                                 | Outputs class probabilities for binary/multiclass classification        |
| Linear Regression     | `LinearRegression()` (scikit-learn)                                   | Fits a line through the data to predict a continuous variable           |
| Train a model         | `.fit(X_train, y_train)` (scikit-learn)                               | Trains the chosen model on training data                                |
| Log Loss              | `log_loss()` (scikit-learn)                                           | Measures prediction error for classification                            |
| Mean Squared Error    | `mean_squared_error()` (scikit-learn)                                 | Measures prediction error for regression                                |
| Hyperparameter tuning | `GridSearchCV()` (scikit-learn)                                       | Test combinations of hyperparameters to find the best performing values |

**Example:** Let's say you're building a model to predict whether a customer will churn. You might start with logistic regression since it's a binary classification problem. You'd train it using log loss to measure performance, adjust its weights with gradient descent, and tune hyperparameters like the regularization strength to avoid overfitting to quirks in your specific customer dataset.

**Think about:**

- Why this model over another? What tradeoffs (interpretability, training time, accuracy) led you there?
- How would you know if the model is underfitting or overfitting during training, and what would you do about each?
- What assumptions does your chosen model make about the data, and do those assumptions actually hold here?

## Evaluation
---
**Core Concept:** Finishing training just means you have _a_ model. Whether it's a _good_ one is what this stage figures out. Evaluation is where you actually check if the model is good, and tune it until it is. This goal is to measure performance objectively and use that measurement to improve the model.

**Splitting for Validation**

| Method                  | What It Does                                                                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Holdout method          | Split data once into training and validation sets. Simple and works well with large datasets.                                                                                                                         |
| K-fold cross-validation | Split data into k folds, train on k-1 of them and validate on the remaining one, repeat until every fold has been the validation set once, then average the results. More reliable, especially with smaller datasets. |
![[Impact Project: Understanding the ML Lifecycle/Images/data split image.png]]
*Source: [Harksys]([URL](https://harksys.com/blog/splitting-data-into-train-validation-and-test-sets/))*
**Model Selection**

Choosing the best model doesn't stop on the highest scorer, it involves:

| Step                     | What It Means                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Out-of-sample validation | Checking performance on data the model wasn't trained on                                                                          |
| Feature selection        | Removing features that aren't beneficial, which can reduce cost and improve performance                                           |
| Hyperparameter tuning    | Searching for the best configuration, usually through grid search (systematic) or random search (random sampling of combinations) |

**Classification Metrics**

Accuracy alone is misleading, you must consider these metrics too:

| Metric    | What It Tells You                                                                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy  | Correct predictions divided by total predictions                                                                                                                                             |
| Precision | Of predicted positives, how much was actually correct. Consider when false positives are costly.                                                                                             |
| Recall    | Of actually positive, how much did the model catch. Consider when false negatives are costly.                                                                                                |
| AUC-ROC   | Compares model performance across every possible classification threshold, not just one. Good for comparing models but doesn't tell you the best threshold for your specific cost tradeoffs. |

Choosing a threshold depends on what's more costly: missing a positive case, or wrongly flagging a negative one. Adjust based on the consequences in your problem.

**Notes:**

- Evaluation is where you find out if all the earlier stages paid off.
- No one metric tells the whole story, the right metric depends on what mistakes are more costly for your problem.
- This stage typically loops back to modeling, tuning is iterative.

**Example:** A fraud detection model. Missing actual fraud (false negative) is expensive, so recall matters more than precision here. You'd rather flag a couple false alarms than let real fraud through. You'd tune the threshold low and use recall as your primary evaluation metric.

**Think about:**

- Which metric matters most for your problem? 
- When would you use holdout vs. k-fold? 
- How would you decide on a threshold in production?
- How could evaluation results change your earlier decisions?

## Deployment
---
**Core Concept:** This is where your model leaves the notebook. Deployment is making the trained model available so it can generate predictions for real use.

**Two Ways to Serve Predictions**

| Method           | What It Does                                                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Batch inference  | Process large volumes of data on a schedule, store the predictions for later use. Lower infrastructure demands, but not real-time. |
| Online inference | Process data as it comes in, real time. Faster response, but harder to implement and more computationally demanding.               |

**Where to Host**

|Option|Tradeoffs|
|---|---|
|Internal hosting|More control and security, but costly and harder to scale.|
|Cloud services|Easier to scale, lower maintenance, but can get expensive at large scale and may raise security concerns.|

**Packaging:** once a hosting method is chosen, the model gets packaged with its dependencies, often using tools like Docker, so it's reproducible and easy to deploy consistently.

Monitoring doesn't stop at deployment. Watch for:

| Risk           | What It Means                                                     |
| -------------- | ----------------------------------------------------------------- |
| Model drift    | Performance degrading as real data shifts away from training data |
| Bias           | The model underperforming for groups it wasn't well trained on    |
| Security risks | Predictions being manipulated by bad actors over time             |

**Notes:**

- Deployment decisions depend on time needs, cost, and security requirements.
- A model isn't "done" once deployed, it needs ongoing monitoring and eventual retraining.
- This stage closes the loop back into Iteration, deployed models get revisited as data and conditions change.

**Example:** A recommendation model for an e-commerce site would likely use online inference (recommendations need to update in real time as a user browses) hosted on the cloud for scalability, packaged in a container for consistency, and monitored for drift as customer behavior shifts seasonally.

**Think about:**

- How would the model be used in production, does that change how you built it? 
- What could go wrong between your validation environment and production?
- How would you roll back if the model performs poorly in the real world? 
- Who's responsible for the model once it's live? 
## Iteration
---
**Core Concept:** The ML lifecycle is a loop. Iteration is where everything you learned from deployment and monitoring feeds back into earlier stages, business understanding gets revisited, new data gets collected, features get reworked, and the model gets retrained. A model is never truly final.

**What triggers iteration:**

|Trigger|What It Means|
|---|---|
|Model drift|Real-world data has shifted away from what the model was trained on, performance quietly degrades|
|New business goals|The organization's objective changes, so the original label or features may no longer be the right target|
|New data availability|More or better data becomes available that could improve performance or fix blind spots|
|Discovered bias or failure cases|Monitoring reveals the model underperforms for a specific group or scenario not caught in evaluation|

**Where iteration loops back to:**

- **Back to Data Preparation** — new data needs cleaning, or a feature needs to be re-engineered
- **Back to Modeling** — retraining with updated data, or trying a different algorithm entirely
- **Back to Evaluation** — re-checking metrics and thresholds against new, current data
- **Back to Business Understanding** — the original problem framing itself no longer fits the business's needs

**Notes:**

- A model is never "done," it's maintained.
- The trigger for iteration should come from monitoring evidence, not a fixed calendar, although scheduled retraining is common practice.

**Example:**

Going back to the e-commerce recommendation model, imagine customer behavior shifts every holiday season, browsing patterns, purchase timing, and product interest all change. Monitoring catches recommendation quality dropping in November. Instead of treating this as a one-time bug, the team retrains the model on recent seasonal data, potentially adding a "time of year" feature. The updated model gets re-evaluated, then redeployed, closing the loop.

**Think about:**

- How would you know if it's time to iterate? Would you rely on scheduled retraining, a performance threshold, or a triggered alert from monitoring? 
- Does every iteration require going back to the beginning? 
- How would you avoid iterating forever without ever shipping?
- At what point is a model "good enough" to redeploy, versus needing another pass?
- Who decides when the business objective itself has changed?


*This guide reflects concepts covered in the Break Through Tech ML curriculum, and official documentation for the libraries referenced.*

