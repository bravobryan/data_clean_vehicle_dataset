# Data Cleaning and Exploratory Data Analysis Project
## The Vehicle Dataset
A project to perform data cleaning tasks, such as detection, treatment, transformation, and data visualization.

The "Vehicle Dataset 2024" dataset was found through Kaggle.com and is available under the ODC Attribution License. https://www.kaggle.com/datasets/kanchana1990/vehicle-dataset-2024
## Dataset Variables

| Variable Name      | Data Type   | Description                                                                   | Variable Example                                    | 
| ------------------ | ----------- | ----------------------------------------------------------------------------- | --------------------------------------------------- |
| **name**           | Text        | The vehicle's full name, including make, model, and trim.                     | `"2024 Jeep Wagoneer Series II"`                    |
| **description**    | Text        | A brief vehicle description, often including key features and selling points. | `"White Knuckle Clearcoat 2023 Dodge Durango Pur…"` |
| **make**           | Categorical | The vehicle manufacturer (Ford, Toyota, BMW).                                 | `"Jeep"`                                            |
| **model**          | Categorical | The model name of the vehicle.                                                | `"Wagoneer"`                                        |
| **type**           | Categorical | The type of vehicle (New, Used).                                              | `"New"`                                             |
| **year**           | Continuous  | The year the vehicle was manufactured.                                        | `2024`                                              |
| **price**          | Discrete    | The price of the vehicles in USD.                                             | `74600.0`                                           |
| **engine**         | Text        | Details about the engine, including type and specifications                   | `"24V GDI DOHC Twin Turbo"`                         |
| **cylinders**      | Discrete    | The number of cylinders in a vehicle's engine.                                | `6.0`                                               |
| **fuel**           | Categorical | The vehicle's fuel type. (Gasoline, Diesel, Electric)                         | `"Gasoline"`                                        |
| **mileage**        | Continuous  | The vehicle's mileage.                                                        | `32.0`                                              |
| **transmission**   | Categorical | The transmission type. (Automatic, Manual)                                    | `"8-Speed Automatic"`                               |
| **trim**           | Categorical | The trim level of the vehicle, indicating different feature sets or packages  | `"Series II"`                                       |
| **body**           | Categorical | The vehicle's body style. (SUV, Sedan, Pickup Truck)                          | `"SUV"`                                             |
| **doors**          | Discrete    | The number of doors on the vehicle                                            | `4.0`                                               |
| **exterior_color** | Categorical | The exterior color of the vehicle                                             | `"White"`                                           |
| **interior_color** | Categorical | The interior color of the vehicle                                             | `"Global Black"`                                    |
| **drivetrain**     | Categorical | The vehicle's drivetrain. (All-wheel Drive, Front-wheel Drive)                | `"Four-wheel Drive"`                                |

# Data Cleaning
## Cleaning Findings
- **Duplicate Row Findings**
    - After removing the `description` column, 31 duplicate rows were found in the dataset.
- **Missing Value Findings**
    - **223** missing values were found in the `price`, `engine`, `cylinders`, `fuel`, `mileage`, `transmission`, `trim`, `body`, `doors`, `exterior_color`, and `interior_color` columns.
    - Count of missing values per column:
        - price: 23 missing,
        - engine: 2 missing,
        - cylinders: 103 missing,
        - fuel: 7 missing,
        - mileage: 32 missing,
        - transmission: 2 missing,
        - trim: 1 missing,
        - body: 3 missing,
        - doors: 7 missing,
        - exterior_color: 5 missing, and
        - interior_color: 38 missing.
    - Data Sparsity of < 2%.
 ![figure01.png](./images/figure01.png)
**Source Code**: _Bilogur, (2018). Missingno: a missing data visualization suite. Journal of Open Source Software, 3(22), 547, https://doi.org/10.21105/joss.00547_
missing.

- **Outlier Findings**
    - The quantitative variables inspected for outliers were the `price`, `cylinders`, `mileage`, and `doors` columns.
- `price`: 28 outliers found, right-skew distribution.
     
![figure02.png](./images/figure02.png)

- `cylinders`: 104 outliers found, left-skew distribution.

![figure03.png](./images/figure03.png)

- `mileage`: 112 outliers found, right-skew distribution.

![figure04.png](./images/figure04.png)

- `doors`: 47 outliers found, left-skew distribution.

![figure05.png](./images/figure05.png)


## Mitigation Methods
- Dropped the `description` column
    - A new DataFrame named `vehicle_only_df` was created by dropping the description column, which is best suited for analysis using NLP techniques and will be excluded from the DataFrame.

## Summary of Outcomes
## Export Clean Dataset

# Limitations
## Impact of Limitations

# Sources
- Bilogur, (2018). Missingno: a missing data visualization suite. Journal of Open Source Software, 3(22), 547, https://doi.org/10.21105/joss.00547
- Kanchana1990. (2024, May 29). Vehicle Dataset 2024. _Kaggle.com_. Retrieved May 31, 2024, from https://www.kaggle.com/datasets/kanchana1990/vehicle-dataset-2024