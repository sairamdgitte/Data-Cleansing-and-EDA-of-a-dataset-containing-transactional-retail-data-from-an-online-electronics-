# Data-Cleansing-and-EDA-of-a-dataset-containing-transactional-retail-data-from-an-online-electronics-
Performing graphical EDA methods to understand the data and then finding and fixing the data anomaly (syntactic, semantic and coverage) Dirty data is identified and fixed, Outliers are identified and removed (Visualized before and after on a boxplot), Missing values are imputed with ML techniques, Calculating the nearest warehouse from the customer's location to deliver the electronic goods ordered (using haversine function) A total of 16 columns containing object and numeric datatypes were treated on.

# Overview
The dataset contains transactional retail data from an online electronics store (DigiCO) located in Melbourne, Australia. The store operation is exclusively online, and it has three warehouses around Melbourne from which goods are delivered to customers.

# Skills that are demonstarted in this project
## Customer review analysis
- By using the sentimental intensity analyzer we first calculate the compound scores:
    if the compound score is greater than or equal to 0.05, then consider the review to be positive
    if the compound score is less than 0.05, then consider the review to be negative
- Found a 'None' component in the reviews column; replaced sentiment of 'None' with True since sentiment analyzer categorises 'None' to be negative (False)

## Date correction
- Split the date on '-' so that we get each year, month and day separately. 
- Make sure the first element is the year (2019) and not month or day. 
    if a month or a day is found, swap it with the year
    if the second element of the splitted date is greater than 12, then swap it with the last element 
- Join the splitted date on '-' and update the df with this value

## Nearest_warehouse and distance_to_nearest_warehouse
  ### Check if there are only 3 values for the warehouse column (which should be the ideal case)
  - We find there are 6 unique values in this column
  - But the error here is that the names of the ware houses are the same but there is a mismatch in their case 
  - Replacing the smaller case names with the camel case names (E.g. 'bakers' is replaced with 'Bakers')

  ### Distance_to_nearest_warehouse
  - Defining a function to calculate the haversine distance between two coordinates
  - Reading in the warehouse's csv to get the details of each warehouse
  - Calculate distance of each customer location with each of the three warehouses
  - Then calculate which is the minimum distance and its indices
  - Loop through the warehouse_csv file and use the above mentioned indices to fetch the name of the warehouse
  - Update the dataframe with the new values found

## Lat Long correction
> By default the values of a Latitude ranges from -90 to 90 and that of Longitude ranges from -180 to 180. But few of the rows have interchanged values for their Lat and Long. Swapping those values should be done in order to correct the errors in these two columns. 

## Seasons
> Seasons have unmatched case. E.g "Summer" and "summer" etc. Also, for few of the rows the dates and the seasons do not match.

  - Splitting the date string to get each component individually (year-month-day)
  - Based on the month number in the date string, comparing the seasons and months as shown in link: http://www.bom.gov.au/climate/glossary/seasons.shtml
  - Replacing the appropriate season names with Camel case formatting

## Order price and order total
> In this method, the item name in shopping_cart will act like 'variables' and no. of items will act as the 'coefficients'. And the entire list of items shopped by a customer will act like an equation which will be used for linear algebra computation 

  1. Creating a new column (df1['shopping_cart_dict']) containing the shopping cart values but in dictionary format

  2. Sorting the column df1['shopping_cart_dict'] inorder to get similar shopping_cart values together

  3. Unzipping the shopping cart tuple and creating two new columns:-
      - crazy_df['coeff'] --> which will have only the coefficients of the equation(shopping cart)
      - crazy_df['variables'] --> which will have only the variables(items) of the equation(shopping cart)
  4. Create a new dataframe which will have all the items of a shopping cart and their respective number of occurances 
  5. Merge this dataframe with crazy_df to get the number of equations present and have it in a column called 'value'
  6. Create another column named ('variable_count') to hold the number of variables in an equation
  7. Drop any row which has a null value in the column 'delivery_charges' (since it cannot be used to calculate the variable value using linalg function)
  8. Filter out this dataframe into chunks of dataframes which are square in nature (in other words keep only those dataframes whose number of equations and number of variables are the same!) and append it to a list.
      - This list will have multiple square dataframes as each element 
  9. Calculate the price of each product using linalg from numpy and store it in a dictionary:
      - This dictionary will have prices for each product (no. of products is 10)
  10. Using this dictionary and the items in the shopping cart, the order price can be calculated. 
  11. Using the calculated order_price, the order_total can be calculated

# Missing Data
## Treating missing values in nearest warehouse column
#### Steps
1. From the original dataframe filter out the rows for which nearest warehouse is NaN and save it in a new dataframe (empty_warehouse) 
2. For each of the customer calculate the distance between all the three warehouses using the haversine function and take it in a list
3. From the above list select the minimum distance among the 3 distances and noting their respective indices
4. Looping through the warehouses.csv file using the above indices to fetch the names of the warehouses and updating it in the missing column

5. Updating the original dataframe using the sub-dataframe (empty_warehouse) mentioned above

## Handling missing values in Latest review
#### Steps
1. Import SentimentIntensityAnalyzer from nltk.sentiment.vader
2. Create a SentimentIntensityAnalyzer() object
3. For all the reviews in the column "latest_customer_review" calculate the compound value:
    - if compound is greater than or equal to 0.05 then flag it as '1'
    - if compound is less than 0.05 then flag it as '0'
 
## Delivery charges missing value
> To predict the missing delivery charges we need to train a model based on the seasons since each season has its own calculation for determining the delivery charges. Th following steps are taken in order to train the model and predict the values.

  ### Steps:
  1. Filter out all the non null values in delivery charges column so that the model we train does not have any missing values in them
  2. For each of the seasons, filter out the respective seasons from the dataframe into a new dataframe
  3. Use LinearRegression package from sklearn.linear_model in-order to train the model for each season
  4. Dependent variable for our model will be delivery_charges since we are supposed to predict the missing values in delivery charges eventually 
  5. Independent variables must be 'distance_to_nearest_warehouse', 'is_expedited_delivery', 'is_happy_customer' since delivery charges depend on these attributes 
  6. Create a model 
  7. Filter out the missing values in column delivery charges in the original dataframe and again filter out the season within it
  8. Extract the indices of each missing value and use the same in the original dataframe to predict the missing values

## Handling missing values in order_price and order_total
  #### Steps:
    In this method, the item name in shopping_cart will act like 'variables' and no. of items will act as the 'coefficients'. And the entire list of items shopped by a customer will act like an equation which will be used for linear algebra computation 

  1. Creating a new column (df1['shopping_cart_dict']) containing the shopping cart values but in dictionary format

  2. Sorting the column df1['shopping_cart_dict'] inorder to get similar shopping_cart values together

  3. Unzipping the shopping cart tuple and creating two new columns:-
      - crazy_df['coeff'] --> which will have only the coefficients of the equation(shopping cart)
      - crazy_df['variables'] --> which will have only the variables(items) of the equation(shopping cart)
  4. Create a new dataframe which will have all the items of a shopping cart and their respective number of occurances 
  5. Merge this dataframe with crazy_df to get the number of equations present and have it in a column called 'value'
  6. Create another column named ('variable_count') to hold the number of variables in an equation
  7. Drop any row which has a null value in the column 'delivery_charges' (since it cannot be used to calculate the variable value using linalg function)
  8. Filter out this dataframe into chunks of dataframes which are square in nature (in other words keep only those dataframes whose number of equations and number of variables are the same!) and append it to a list.
      - This list will have multiple square dataframes as each element 
  9. Calculate the price of each product using linalg from numpy and store it in a dictionary:
      - This dictionary will have prices for each product (no. of products is 10)
  10. Using this dictionary and the items in the shopping cart, the order price can be calculated. 
  11. Using the calculated order_price, the order_total can be calculated


# Outlier Data
## Train a model to predict the delivery_charges
1. delivery_charges depends on 'is_expedited_delivery', 'distance_to_nearest_warehouse' and 'is_happy_customer' attributes
2. Predict the values
3. After predicting the values subtract it from the original value
4. Calculate the z score of the column values 
5. Consider the lower and upper to be 2 sigma and anything below or above this will be an outlier
6. Filter out the data for which the zscore column is 0
