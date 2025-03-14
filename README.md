# Tariff Impact Analysis API Dashboard

This project is an **ETL pipeline and API dashboard** designed to analyze the impact of tariffs on historical market data. The application is built with **Flask**, utilizes **PostgreSQL** for data storage, processes data with **Python scripts**, and presents insights via a **D3.js front-end dashboard**.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Tariff Impact Analysis API Dashboard provides a comprehensive analysis of how tariffs affect historical market data. It includes:
- A **Flask-based REST API** to serve tariff impact data.
- **Python scripts** for fetching and processing market data and tariff events.
- A **PostgreSQL database** to store processed data.
- An interactive **D3.js dashboard** for visualizing insights.

---

## Architecture

The project consists of the following components:

- **Flask App**: REST API to serve tariff impact data.
- **Historical Market Python Script**: Fetches and processes market data.
- **Tariff Event Script**: Collects and processes tariff-related events.
- **PostgreSQL Database**: Stores market data and tariff event records.
- **Dockerized PostgreSQL Container**: Deploys the database in a containerized environment.
- **D3.js Dashboard**: Interactive front-end visualization for tariff analysis.

---

## Project Structure
