B1. Problem Formulation

(a) ML Problem Formulation

In this case, we want to predict how many items will be sold in a store for a given promotion and month.

Target Variable (what we want to predict):
* Number of items sold (sales volume)

Input Features (what we use to predict):
* Type of promotion (Flat Discount, BOGO, etc.)
* Store location type (urban / semi urban / rural)
* Store size
* Monthly footfall
* Competition density nearby
* Customer demographics (age group, income etc.)
* Month / season (festival time, off season etc.)


Type of ML Problem:
* This is a Regression problem because the target (items sold) is a continuous numerical value.

Justification:
* We are not predicting categories (like yes/no), but an actual number (like 500 items, 1200 items). So regression is the correct choice.

(b) Why Items Sold is Better than Revenue

Using total sales revenue can be misleading in this situation.
* Revenue depends on price of items, not just promotion effectiveness
* For example:
    * A high-price item sold less quantity can still give high revenue
    * A promotion may increase number of items sold but reduce price which is revenue may not increase much

So revenue does not clearly show how effective a promotion is.

On the other hand, items sold (volume) directly shows:
* How many products customers are buying
* How well the promotion is attracting customers

In ML projects, we should always choose a target variable that directly reflects the business goal.
Here, the goal is to increase product movement, so items sold is the correct target, not revenue.

(c) Better Modelling Strategy than Single Global Model

Using one single model for all 50 stores is not a good idea because:
* Stores in different locations behave differently
* Customer preferences vary (urban vs rural)
* Promotions may work differently in each area


We can use a segmented modelling strategy, like:
* Group stores based on type:
    * Urban stores --> one model
    * Semi-urban stores --> one model
    * Rural stores --> one model
OR
* Use a Hierarchical / Mixed Model where:
    * There is a global model
    * But it also learns store-specific behaviour

Justification:
This approach captures local differences, so predictions will be more accurate.
It avoids the mistake of assuming all stores behave the same, which is not true in real life.



B2. Data and EDA Strategy

(a) Data Joining and Final Dataset Design

We have 4 tables:
* Transactions table
* Store attributes table
* Promotion details table
* Calendar table

How to join:
* Start with transactions table (main data)
* Join with store attributes using store_id
* Join with promotion details using promotion_id
* Join with calendar table using date

So finally, all data will be in one combined table.

Grain of final dataset: 
One row = one store + one month + one promotion

(we aggregate daily transactions into monthly level because promotions are monthly)

Aggregations to perform:

From transactions table:
* Total items sold per store per month
* Total number of transactions
* Average basket size (items per bill)

From calendar:
* Number of weekends in that month
* Festival count in that month

Other features:
* Promotion type (categorical)
* Store features (size, location, etc.)

Raw transaction data is too detailed (bill-level), but decision is taken at monthly store level, so we aggregate to match business use.

(b) EDA (Exploratory Data Analysis)
Before building model, we should understand the data properly.

1. Sales Distribution Plot
    * Check if data is skewed or normal
    * Look for extreme values 

Impact:
If highly skewed - may apply log transformation
If outliers - may cap or handle separately

2. Promotion Type vs Items Sold
    * Compare how each promotion performs

Look for:
    * Which promotion gives higher median sales
    * Variability in performance
Impact:
    * Helps in feature importance understanding
    * May create interaction features later

3. Store Type vs Sales (Urban vs Rural comparison)
Compare sales across location types

Look for:
    * Do urban stores sell more?
    * Does promotion effect differ by location?

Impact:
    * May need separate models or interaction features


4. Time-based Trend (Line chart month vs sales)
* Check seasonality

Look for:
    * Festival spikes
    * Monthly patterns

Impact:
    * Add time features (month, festival flag)
    * Helps model capture seasonality


5. Correlation Heatmap (optional but useful)
* Check relation between numerical features

Look for:
    * Strong correlations (multicollinearity)

Impact:
    * Remove redundant features
    * Improve model stability


(c) Handling Imbalance (80% No Promotion Data)

Problem:
* Most data is from no promotion cases
* Model may learn that “no promotion” is normal and ignore promotion impact

Effect on model:
* Poor learning of promotion effectiveness
* Biased predictions

Steps to handle:
* Resampling:
    * Oversample promotion data
    * Or undersample no-promotion data
* Stratified sampling:
    * Ensure training data has good mix of all promotion types
* Feature engineering:
    * Add a flag: promotion vs no promotion
* Model choice:
    * Use models that can handle imbalance better (like tree-based models)

Make sure model properly learns promotion impact, not just default behaviour.


B3. Model Evaluation and Deployment
(a) Train-Test Split and Evaluation Metrics

We have 3 years of monthly data, so this is clearly time-based data.
How to split:
* Use time-based split, not random
Example:
    * Train data → First 2 years
    * Test data → Last 1 year

OR
* Use rolling validation

Why random split is wrong:
* Random split will mix past and future data
* Model may learn from future data → data leakage
* In real life, we always predict future using past, so split should follow time order

Evaluation Metrics:
1. MAE (Mean Absolute Error)
    * Tells average error in number of items
    * Easy to understand
    * Example: MAE = 50 means prediction is off by ~50 items
2. RMSE (Root Mean Squared Error)
    * Gives more penalty to large errors
    * Useful if big mistakes are costly

3. R² Score (R-squared)
    * Tells how much variance is explained by model
    * Value closer to 1 is better

Business Interpretation:
MAE  - how wrong we are in actual sales units
RMSE - helps avoid big wrong decisions
R^2  - overall model quality

(b) Explaining Different Recommendations (Feature Importance)

Model is recommending:
    * December → Loyalty Points Bonus
    * March → Flat Discount

Even though same store, recommendation is different because input features are different.

How to investigate:
* Use feature importance (global level)
    * See which features matter most (month, promotion type, footfall, etc.)
* Use local explanation (for that prediction)
    * Tools like SHAP values (if used)

What we may find:
* December:
    * Festival season
    * High customer spending
    * Loyalty points may work better
March:
    * No major festivals
    * Customers more price-sensitive
    * Flat discount works better

How to explain to marketing team:
“In December, customers are already buying more due to festivals, so giving loyalty points increases repeat purchase.”
“In March, customers are more price conscious, so direct discount works better.”

Model is not random, it is using season + store + customer behaviour to decide.



(c) Deployment Process

We want model to run every month without retraining.

Step 1: Save the Model
* After training, save model using:
    * joblib or pickle
* Also save preprocessing steps (scaling, encoding etc.)

Step 2: Prepare New Monthly Data
* At start of each month:
    * Collect latest data:
        * Store features
        * Promotion options
        * Calendar info
    * Apply same preprocessing as training
* Create input dataset:
    * One row per store + promotion combination

Step 3: Generate Predictions
* Load saved model
* Predict items sold for each promotion
* Select promotion with highest predicted sales

Step 4: Output Recommendation
* For each store:
    * Choose best promotion
* Share with marketing team

Step 5: Monitoring Model Performance
We should continuously check if model is still working well.
What to monitor:
1. Prediction vs Actual sales
    * Track error (MAE over time)
2. Data Drift
    * Check if input data distribution has changed
    * Example: customer behaviour changed
3. Performance Drop
    * If error increases consistently - model is outdated

Step 6: Retraining Strategy
* Retrain model:
    * Every few months OR
    * When performance drops

Overall idea:
    * Build once
    * Use every month
    * Monitor continuously
    * Retrain when needed