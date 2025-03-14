# Tariff Impact Analysis API Dashboard

This project is an **ETL pipeline and API dashboard** designed to analyze the impact of tariffs on historical market data. The application is built with **Flask**, utilizes **PostgreSQL** for data storage, processes data with **Python scripts**, and presents insights via a **D3.js front-end dashboard**.

---

## Key Performance Indicators (KPIs)

The following KPIs are tracked and visualized in the D3.js dashboard:

| KPI                          | Description                                      |
|------------------------------|--------------------------------------------------|
| **Market Impact Score**       | Measures the overall impact of tariffs on market data. |
| **Tariff Event Frequency**    | Number of tariff-related events over time.       |
| **Price Change Percentage**   | Percentage change in market prices due to tariffs.|
| **Affected Industries**       | Industries most impacted by tariff changes.      |

---

## JavaScript Code Snippets

Below are some key JavaScript snippets used in the D3.js dashboard:

### 1. Fetching Data from the Flask API

```javascript
async function fetchMarketData() {
    const response = await fetch('/api/market-data');
    const data = await response.json();
    return data;
}
