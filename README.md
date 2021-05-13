# Predicting building electricity consumption using KNN

This was originally a project made possible thanks to a Coursera Guided Project. The goal is to use K-Nearest Neighbors to predict the electricity consumption of buildings based on past use. I decided to transform it into a Streamlit dashboard where you can filter by timezone and industry.

The project also only used K=1 for simplicity but my dashboard lets you choose up to K=6, plots the actual vs predicted scatter plot, and also gives the R-square value.

I have one main struggle that I want to work on. For the test, train split, rather than randomly splitting the data 70-30, the split is done truncating data up to a certain date and then predict the values for the other years. My struggle is trying to use a streamlit widget that I can use to change the dates more dynamically. It probably won't add a lot to the project, but I want to get familiar with working with datetime structures in Streamlit.

Future ideas:

- Use different algorithms to make the prediction and compare them.
- Allow the user to choose the algorithm and have the dashboard display the results.

This dataset does have one major limitation: not every combination of timezone and industry has enough data to complete all parts of the analysis, if any at all.
