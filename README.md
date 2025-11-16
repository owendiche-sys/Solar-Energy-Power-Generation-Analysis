\# Solar Power Generation Prediction



This project predicts \*\*solar power generation\*\* based on weather and environmental features using machine learning. It demonstrates data preprocessing, exploratory data analysis, modeling, evaluation, and feature importance analysis for renewable energy applications.





---



\## Project Overview



The goal is to predict solar energy output using environmental factors. This helps in optimizing solar panel performance and forecasting energy generation.



Key steps:



&nbsp;\*\*Data Exploration\*\*

&nbsp;  - Inspect dataset shape and data types.

&nbsp;  - Check for missing values.

&nbsp;  - Visualize the distribution of generated power.



&nbsp;\*\*Data Preprocessing\*\*

&nbsp;  - Separate features (`X`) and target (`y`).

&nbsp;  - Train/test split (80/20).



&nbsp;\*\*Modeling\*\*

&nbsp;  - Random Forest Regressor trained on weather features.

&nbsp;  - Prediction on test set.



&nbsp;\*\*Evaluation\*\*

&nbsp;  - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score.

&nbsp;  - Visual comparison of actual vs predicted generated power.



&nbsp;\*\*Feature Importance\*\*

&nbsp;  - Analyze which features most influence solar power output.

&nbsp;  - Visualize using a horizontal bar plot.



---



\## Results



\- \*\*Distribution of Generated Power:\*\*  

&nbsp; !view result folder



\- \*\*Actual vs Predicted Power:\*\*  

&nbsp; !\[Actual vs Predicted](results/actual\_vs\_predicted.png)



\- \*\*Feature Importance:\*\*  

&nbsp; !\[Feature Importance](results/feature\_importance.png)  

&nbsp; Features like \*\*temperature, cloud cover, and wind speed\*\* were most influential.



\- \*\*Performance Metrics:\*\*

&nbsp; - Mean Absolute Error (MAE): \*253.78\*

&nbsp; - Root Mean Squared Error (RMSE): \*404.10\*

&nbsp; - R² Score: \*0.82\*



---





