## Project description & motivation
The goal of this project is to develop a machine learning model that uses real estate listing data to predict Berlin apartment rent prices (Kaltmiete).
Kaltmiete was chosen as the goal variable to ensure an accurate dataset because Warmmiete (rent including utilities) is not always available across listings.
The information was gathered by collecting publicly accessible listings from Immowelt, a significant German real estate marketplace. This source was chosen because it provides a large number of listings and rich property features such as apartment size, number of rooms, amenities, and building characteristics.
The data collected makes it possible to create and evaluate machine learning models that predict rental costs based on property characteristics.

## Legal & ethical considerations
Since the website doesn’t provide any APIs to access the data, it was necessary to write a script to scrape the data. While writing the script every aspect of legal considerations in Germany, such as GDPR, was considered. Personal data like the phone number and other data of the person or the company who posted the listing weren’t scraped. These data are sensitive and aren’t needed for the model. A derived variable, such as whether the poster is the owner or a real estate agency might have had some correlation with the price, but it would require sensitive data, and it is also hard to determine. 
The scraped data is publicly available, and this project has no commercial application, but rather serves as a personal project only for learning and demonstration purposes. 
Content protected by copyright like articles and images were not scraped. 
Security bypassing mechanisms and other harmful and forbidden mechanisms were not implemented.
- The link that we used is “https://www.immowelt.de/classified-search?distributionTypes=Rent&estateTypes=House,Apartment&locations=AD08DE8634&page={pageNumber}”, which according to the robots.txt file we are allowed to access this link via web scrapers.
- Also to be respectful and not to flood the website with traffic, commands like time.sleep() were used to slow down the scraping script.

## Data collection
When it comes to the number of samples that we are going to scrape for a regression task, I would suggest at least 20,000 samples as a sufficient amount to collect, considering that we might lose up to 40% of our dataset during data preparation, so 12,000 samples would be enough to build a decent model.
But since Berlin is a very large city, I would aim for at least 25,000 scrapes.
The script scrapes the following data:
1.	Listing URL – string (unique identifier for each listing)
2.	Cold rent (Kaltmiete) – continuous (string, numeric value embedded, requires parsing)
3.	Warm rent (Warmmiete) – continuous (string, numeric value embedded, may be missing)
4.	Deposit (Kaution) – continuous (string, numeric value embedded)
5.	Area – continuous (string, square meters, requires parsing)
6.	Number of rooms – continuous (string, requires parsing)
7.	Floor – categorical (string, e.g., “3. OG”, “EG”, “DG”)
8.	Free from – temporal (string, requires formatting to date or category)
9.	Location / Address – string
10.	Has balcony – boolean
11.	Has terrace – Boolean
12.	Has garden – boolean
13.	Has elevator – boolean
14.	Has parking – boolean
15.	Has cellar (Keller) – boolean
16.	Barrier-free access – Boolean
17.	Has fitted kitchen (Einbauküche) – boolean
18.	Has bathtub – boolean
19.	Has shower – boolean
20.	Flooring type – categorical (string)
21.	Energy source – categorical (string)
22.	Heating type – categorical (string)
23.	Property condition – categorical (string)
24.	Year built – continuous (string, integer year)
25.	Energy certificate – categorical (string)
26.	Energy demand – continuous (string, e.g., “x kWh/m²a”)
27.	Schufa check required – boolean
28.	Number of images posted – discrete (string, integer)
29.	Scraped at – datetime (string, timestamp)
