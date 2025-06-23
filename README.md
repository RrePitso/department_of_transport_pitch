# 🚌 Public Transport AFC Anomaly Detection Dashboard
A Streamlit-based interactive dashboard that simulates and detects anomalies in public transport AFC (Automated Fare Collection) transaction data. Inspired by the MyCiTi and Rea Vaya systems in South Africa, this tool showcases how data science can uncover operational issues, revenue leakage, and system inefficiencies in real-time transit fare systems.

🚀Live Dashboard
https://departmentoftransportpitch.streamlit.app/

🚀 Features
Synthetic AFC Transaction Simulation
Generate thousands of realistic smart card transaction records based on public route data.

Anomaly Detection
Identify critical anomalies like:

Missing Tap-Outs

Fare Discrepancies

Taps with Negative Balance

Over-speed Trips (e.g. implausible travel times)

KPI Overview
Get high-level insights into expected vs. actual revenue, anomaly counts, and more.

Visual Insights
Interactive visualizations using matplotlib and seaborn:

Daily/hourly transaction trends

Revenue loss breakdowns

Anomaly distributions over time

Filtering and Exploration
Easily filter transactions by anomaly type or bus route number.

📊 Sample Use Cases
Municipal Transit Authorities: Audit AFC data to identify revenue leakage.

Data Science Students: Learn about simulation, data cleaning, anomaly detection, and Streamlit dashboards.

Transport Planners: Visualize transaction trends and explore data-driven decision-making.

💡 Key Insights
Simulated Data = Infinite Possibility
Since real transit data is hard to access, I built a full synthetic data pipeline using:

Public bus route data from the MyCiTi system

Fare structures for both MyCiTi (off-peak/peak) and Rea Vaya

Randomized trip distances and speeds

Logic to simulate realistic anomalies

Anomaly Injection
Each transaction has a chance to include specific anomalies. This helps simulate real-world scenarios like:

Fare underpayment

Trips with missing exits (tap-outs)

Extremely fast trip speeds (over-speed trips)

Revenue Protection
The dashboard quantifies total potential losses due to anomalies, which can help inform fraud-prevention strategies.

🛠️ Technologies Used
Tool / Library	Purpose
Python	Core programming language
Streamlit	Dashboard UI
Pandas, NumPy	Data manipulation
Matplotlib, Seaborn	Visualizations
datetime, random	Simulation logic
CSV Route File	Real-world data integration (MyCiTi routes)

📁 Project Structure
bash
Copy
Edit
├── app.py
├── Integrated_rapid_transit_(IRT)_system_MyCiTi_Bus_Routes.csv
├── requirements.txt
└── README.md
🧪 How to Run It Locally
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/public-transport-afc-dashboard.git
cd public-transport-afc-dashboard
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
You may need to install matplotlib, seaborn, and streamlit if not already installed.

3. Run the dashboard
bash
Copy
Edit
streamlit run app.py
Make sure the file Integrated_rapid_transit_(IRT)_system_MyCiTi_Bus_Routes.csv is in the same directory.

📍 Notes
All transaction data is synthetically generated.

This project is intended for demonstration and educational purposes.

Fare structures are based on publicly available estimates as of 2025.

👤 Author
Ofentse Pitso
Final-year Computer Science Student | Data Science Enthusiast
LinkedIn • GitHub

📌 Future Improvements
Add clustering or ML-based anomaly detection

Real-time dashboard integration (with APIs)

Improve geospatial visualizations of routes and anomalies
