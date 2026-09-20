## Website

https://berlin-rent-price-evaluation.streamlit.app/

## Project description & motivation

The goal of this project is to develop a machine learning model that predicts Berlin apartment rent prices (Kaltmiete) from real estate listing data.
Kaltmiete was chosen as the target variable because Warmmiete (rent including utilities) is not consistently available across listings.
Data was collected from publicly accessible listings on Immowelt, a major German real estate marketplace, chosen for its large number of listings and rich property features such as size, number of rooms, amenities, and building characteristics.
The resulting dataset supports building and evaluating ML models that estimate rental costs from property attributes.

## Key Highlights / Results

### Dataset

- Total scraped listings: 22,687
- Listings after cleaning: 15,588
- Data source: Immowelt
- Scraping period: 28/12/2025 – 31/01/2026
- Raw features collected: 29
- Final model features: 15
- Missing data handled using: category-specific rules (see cleaning notebook)
- Duplicate listings removed: yes – similarity method
- **K-anonymity generalization** applied to location-related features before any publication; `location` and `listing_url` were dropped from the modeling dataset to reduce re-identification risk.

### Feature Engineering

- Categorical encoding: one-hot encoding
- Numerical preprocessing: standard scaling
- Final dataset samples: 12470 samples for training and 3118 for testing.

### Exploratory Data Analysis – key findings

- Rent distribution is strongly right-skewed.
- Strong correlation between apartment size (m²) and rent price.
- Features most correlated with rent:
  - area (m²)
  - number of rooms
  - year of construction
- Amenities such as balcony, elevator, and parking tend to increase rent.
- Some features showed unexpected or weak effects during EDA (e.g. `has_terrasse`, `has_bathtub`), which motivated a feature-ablation study during modeling.

### Machine Learning Models

The following regression models were implemented and compared:

- Linear Regression
- KNN Regressor
- Support Vector Regressor
- Random Forest Regressor
- XGBoost
- Neural Network

Training strategy:

- Cross-validation: 5-fold
- Hyperparameter tuning: Grid Search
- Feature-ablation experiments: all features / −2 features / −7 features

### Model Performance

Metrics: RMSE, R²

| Model                                             | RMSE   | R²     |
| ------------------------------------------------- | ------ | ------ |
| XGBoost (without 7 least important featres)       | 177.29 | 0.9398 |
| XGBoost                                           | 184.13 | 0.9350 |
| Random Forest (without 7 least important featres) | 204.44 | 0.9199 |
| Random Forest (no bathtub/terrasse)               | 206.31 | 0.9184 |
| Random Forest                                     | 209.19 | 0.9161 |
| Neural Network                                    | 221.42 | 0.9061 |
| Neural Network (no bathtub/terrasse)              | 222.38 | 0.9051 |
| KNN                                               | 265.52 | 0.8649 |
| SVR                                               | 408.33 | 0.6806 |
| Linear Regression                                 | 408.98 | 0.6796 |

> **Final model choice:** Random Forest (200 estimators, depth 40).
> XGBoost had a slightly higher R², but XGBoost Quantile Regression performed poorly and standard XGBoost does not provide a native uncertainty estimate. Random Forest natively provides tree-disagreement–based uncertainty, which was a project requirement. The small accuracy trade-off was accepted in exchange for usable confidence estimates.

### Uncertainty & Confidence

- Random Forest provides an uncertainty estimate from the disagreement between individual trees (`pred_std`).
- Correlation between tree disagreement and absolute prediction error: **0.536** – a moderate-to-strong positive relationship, meaning the model “knows when it doesn’t know.”
- This is exposed to the user as a low / medium / high confidence indicator rather than a single point prediction.

### Model Interpretation

Most influential predictors of rent:

1. Apartment size (m²)
2. Number of rooms
3. Availability (`free_from_unknown`)
4. Year built
5. Floor
6. District (e.g. Mitte)

Explainability methods used:

- Feature importance from tree-based models
- SHAP values (summary + waterfall)
- Partial dependence plots
- Individual-tree prediction breakdown
- Error analysis by district, property condition, heating type, and number of rooms

### Model Card

A model card for the final Random Forest is included in the repository (see `model_card.pdf`), covering intended use, limitations, bias considerations, and evaluation metrics.

### Some insights derived from the analysis

- Rent increases by approximately 15.23 € per additional m².
- Apartments with elevator, parking, or built-in kitchen show higher average rent.
- Districts such as Nikolassee, Grünau, Mitte, Bohnsdorf, and Siemensstadt have significantly higher rent levels.
- Price per m² ranking differs: Oberschöneweide, Blankenburg, Mitte, Dahlem, and Friedenau lead.
- During the Cold War, West Berlin saw significantly more construction than East Berlin.
- No significant price difference between West and East Berlin overall; East Berlin slightly overtakes the West on price per m².
- Split into four regions, the West is the most expensive, followed by the South; North and East are cheaper.
- Splitting by distance to the center: central areas are most expensive; surrounding areas cheaper than suburban areas (likely noise / new construction effect).
- Some features such as wooden floor, hybrid energy source, new or renovated properies are associated with higher prices while unknown values tend to have lower rent prices.

### Example Prediction

Apartment features:

- Size: 87.8 m²
- Rooms: 3
- Floor: 1
- Availability: immediately
- District: Lichtenberg, Berlin
- Built-in kitchen: Yes
- Bathtub: Yes
- Shower: Yes
- Elevator: Yes
- Flooring: Vinyl
- Energy source: Other
- Heating: Underfloor heating
- Property condition: New / first occupancy
- Year built: 2025

Prediction:

- Predicted cold rent: ≈1,793 €
- Actual rent: 1,799 €
- Error: ≈6 €

### Visual Highlights

The repository includes visualizations such as:

- Rent price distribution histogram
- Price vs. area and number of rooms
- Comparison of Berlin regions (N/S/E/W, center vs. surrounding vs. suburbs, East vs. West)
- Correlation heatmap
- Feature importance ranking
- SHAP summary and waterfall plots
- Partial dependence plots
- Prediction vs. actual scatter plot with uncertainty encoding
- Interactive Berlin choropleth map (price per m² by ZIP)
- Interactive scatterplot matrix and dashboard views

<h2 align="center"> Feature Analysis Analysis </h2>

<p align="center">
  <img src="graphs/readme_graphs/price_distribution.png" width="49%" alt="Rent price distribution histogram">
  <img src="graphs/readme_graphs/district_distribution.png" width="49%" alt="District distribution histogram">
</p>

<p align="center">
  <img src="graphs/readme_graphs/year_built_count.png" width="49%" alt="Year built cumulative graph">
  <img src="graphs/readme_graphs/number_of_floors.png" width="49%" alt="Number of floor distribution treemap">
</p>

<h2 align="center"> Relation of different features with rent prices </h2>

<p align="center">
  <img src="graphs/readme_graphs/area_and_price.png" width="49%" alt="Relation of area and price">
  <img src="graphs/readme_graphs/floor_number_and_price.png" width="49%" alt="Floor number and price">
</p>

<p align="center">
 <img src="graphs/readme_graphs/category_vs_price_2.png" width="80%" alt="Feature exploration">
</p>

<p align="center">
  <img src="graphs/readme_graphs/district_vs_price.png" width="49%" alt="Price based on different geographical divisions">
  <img src="graphs/readme_graphs/district_vs_price_3.png" width="49%" alt="Most expensive and affordable districts">
</p>

<p align="center">
 <img src="graphs/readme_graphs/year_and_price.png" width="80%" alt="Year vs price">
</p>

<h2 align="center"> Correlation Analysis </h2>

<p align="center">
  <img src="graphs/readme_graphs/correlations_matrix.png" width="49%" alt="Matrix of correlation between variables">
  <img src="graphs/readme_graphs/scatterplot.png" width="49%" alt="Scatterplot matrix of correlation between variables">
</p>

<h2 align="center"> Historical & Geographic Analysis </h2>

![Prices per meter square for each district of Berlin](graphs/readme_graphs/berlins_map.png)
![Distribution of apartments built during Communism - West vs East](graphs/readme_graphs/east_vs_west.png)
![Distributions of apartments based on the number of rooms through the decades](graphs/readme_graphs/num_of_rooms_vs_decade.png)

<h2 align="center"> Model Insights and performance </h2>

<p align="center">
  <img src="graphs/readme_graphs/shap_values.png" width="49%" alt="SHAP values">
  <img src="graphs/readme_graphs/partial_dependence.png" width="49%" alt="Partial dependence of area to predictied value">
</p>

![Radar chart of performance for the most important models](graphs/readme_graphs/radar_chart.png)

<p align="center">
  <img src="graphs/readme_graphs/error_distribution.png" width="49%" alt="Error distribution">
  <img src="graphs/readme_graphs/error_by_district.png" width="49%" alt="MAE error and bias by district">
</p>

## Legal & ethical considerations

Since the website does not provide any APIs to access the data, a scraping script was necessary. While writing the script, every aspect of legal considerations in Germany, such as GDPR, was considered. Personal data such as phone numbers and other data of the person or company who posted the listing were not scraped. These data are sensitive and not needed for the model. A derived variable such as whether the poster is the owner or a real estate agency might correlate with price, but it would require sensitive data and is hard to determine.
The scraped data is publicly available and this project has no commercial application – it is a personal project for learning and demonstration purposes.
Content protected by copyright such as articles and images was not scraped.
Security-bypassing mechanisms and other harmful or forbidden mechanisms were not implemented.

- The URL used was `https://www.immowelt.de/classified-search?distributionTypes=Rent&estateTypes=House,Apartment&locations=AD08DE8634&page={pageNumber}`, which according to the `robots.txt` file is allowed to be accessed by web scrapers.
- To be respectful and not flood the website with traffic, `time.sleep()` was used to slow down the scraping script.

### Ethics & bias

- **PII:** No names, phone numbers, or poster identities were collected. Full location was collected but generalized via k-anonymity before any publication scenario.
- **Quasi-identifiers:** ZIP code and district are quasi-identifiers; they are kept in the working dataset but would be removed or generalized before public release.
- **Bias review:** Historical, population, self-selection, social, temporal, measurement, representation, aggregation, sampling, and evaluation biases were considered. See the final report ethics section for the full discussion.
- **Fairness:** No protected attributes are present in the dataset.
- **Temporal limitation:** The data is a single scraping window (28/12/2025 – 31/01/2026), so the project could not be treated as a time series.

## Data collection

For a regression task, at least 20,000 samples were targeted, accounting for up to 40% loss during preparation; 25,000+ scrapes were aimed for given Berlin's size.
The script scrapes the following data:

1. Listing URL – string (unique identifier)
2. Cold rent (Kaltmiete) – continuous (string, numeric value embedded, requires parsing)
3. Warm rent (Warmmiete) – continuous (string, may be missing)
4. Deposit (Kaution) – continuous (string, numeric value embedded)
5. Area – continuous (string, m², requires parsing)
6. Number of rooms – continuous (string, requires parsing)
7. Floor – categorical (string, e.g. “3. OG”, “EG”, “DG”)
8. Free from – temporal (string, requires formatting to date or category)
9. Location / Address – string
10. Has balcony – boolean
11. Has terrace – boolean
12. Has garden – boolean
13. Has elevator – boolean
14. Has parking – boolean
15. Has cellar (Keller) – boolean
16. Barrier-free access – boolean
17. Has fitted kitchen (Einbauküche) – boolean
18. Has bathtub – boolean
19. Has shower – boolean
20. Flooring type – categorical (string)
21. Energy source – categorical (string)
22. Heating type – categorical (string)
23. Property condition – categorical (string)
24. Year built – continuous (string, integer year)
25. Energy certificate – categorical (string)
26. Energy demand – continuous (string, e.g. “x kWh/m²a”)
27. Schufa check required – boolean
28. Number of images posted – discrete (string, integer)
29. Scraped at – datetime (string, timestamp)
