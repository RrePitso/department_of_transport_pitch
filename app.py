import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="AFC Data Anomaly Dashboard", page_icon="📈")

st.title("🚇 Public Transport AFC Data Anomaly Detection")
st.markdown("---")

# --- 1. Data Loading and Simulation (Cached for Performance) ---

@st.cache_data # Cache the data loading and initial processing
def load_and_prepare_routes(file_path):
    """Loads the MyCiTi routes CSV and prepares it."""
    try:
        df_routes = pd.read_csv(file_path)
        active_routes_df = df_routes[df_routes['RT_STS'] == 'Active'].copy()
        active_routes_df['route_length_km'] = active_routes_df['SHAPE_Length'] / 1000
        return active_routes_df
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as app.py.")
        st.stop() # Stop the app if file is not found
    except Exception as e:
        st.error(f"Error loading routes data: {e}")
        st.stop()

@st.cache_data # Cache the simulation result as it's computationally intensive
def simulate_transactions(num_trans, routes_df, current_time, sim_days,
                          peak_hours, fares_peak, fares_offpeak):
    """Simulates AFC transactions with injected anomalies."""
    transactions = []
    
    # Ensure routes_df is not empty before sampling
    if routes_df.empty:
        st.warning("No active routes found to simulate transactions. Please check route data.")
        return pd.DataFrame() # Return empty DataFrame

    for i in range(num_trans):
        # Simulate a transaction time within the next 'sim_days'
        transaction_time = current_time + timedelta(seconds=random.randint(0, sim_days * 24 * 60 * 60))

        is_peak = is_peak_hour(transaction_time, peak_hours)

        # Randomly select an active route
        selected_route = routes_df.sample(1).iloc[0]
        route_length_km = selected_route['route_length_km']

        # Simulate a trip distance: 30% to 100% of route length, min 1km, max 50km
        trip_distance_km = max(1.0, min(50.0, route_length_km * random.uniform(0.3, 1.0)))

        calculated_fare = get_fare(trip_distance_km, is_peak, fares_peak, fares_offpeak)

        # Simulate tap-in and tap-out times, assuming trip duration proportional to distance
        tap_in_time = transaction_time
        trip_duration_minutes = max(5, int(trip_distance_km * 2) + random.randint(-5, 15)) # 2 min/km + randomness
        tap_out_time = tap_in_time + timedelta(minutes=trip_duration_minutes)

        # Introduce anomalies for demonstration
        anomaly_choice = random.choices(
            ['none', 'missing_tap_out', 'incorrect_fare', 'negative_balance_tap', 'over_speed_trip'],
            weights=[0.80, 0.08, 0.06, 0.03, 0.03], k=1
        )[0]

        simulated_fare_paid = calculated_fare
        if anomaly_choice == 'incorrect_fare':
            simulated_fare_paid = round(calculated_fare * random.uniform(0.5, 1.5), 2)
        elif anomaly_choice == 'negative_balance_tap':
            simulated_fare_paid = 0 

        transactions.append({
            'transaction_id': i + 1,
            'card_id': f'CARD{random.randint(10000, 99999)}',
            'route_name': selected_route['RT_NAME'],
            'route_number': selected_route['RT_NMBR'],
            'trip_distance_km': round(trip_distance_km, 2),
            'tap_in_time': tap_in_time,
            'tap_out_time': tap_out_time if anomaly_choice != 'missing_tap_out' else None,
            'is_peak_hour': is_peak,
            'calculated_fare': round(calculated_fare, 2),
            'actual_fare_paid': round(simulated_fare_paid, 2),
            'anomaly_type': anomaly_choice,
            'simulated_speed_kph': round(trip_distance_km / (trip_duration_minutes / 60), 2) if anomaly_choice != 'missing_tap_out' else np.nan
        })

    df_trans = pd.DataFrame(transactions)
    # Convert 'tap_in_time' to datetime if it's not already
    df_trans['tap_in_time'] = pd.to_datetime(df_trans['tap_in_time'])
    df_trans['tap_in_date'] = df_trans['tap_in_time'].dt.date
    df_trans['tap_in_hour'] = df_trans['tap_in_time'].dt.hour
    return df_trans

@st.cache_data # Cache the anomaly detection result
def detect_transaction_anomalies(df):
    """Detects anomalies in the simulated transaction data."""
    anomalies_list = []

    if df.empty:
        return anomalies_list

    # 1. Missing Tap-Outs
    missing_tap_out = df[df['tap_out_time'].isnull() & (df['anomaly_type'] == 'missing_tap_out')] # Check specifically for the injected anomaly
    if not missing_tap_out.empty:
        anomalies_list.append({
            'type': 'Missing Tap-Out',
            'description': f"Transactions with missing tap-outs.",
            'count': len(missing_tap_out),
            'transactions_df': missing_tap_out
        })

    # 2. Fare Discrepancies (Actual paid vs. Calculated)
    # Using a small tolerance for float comparisons, and excluding the deliberate 'negative_balance_tap' if we want to separate it
    fare_discrepancy = df[(abs(df['actual_fare_paid'] - df['calculated_fare']) > 0.01) & (df['anomaly_type'] == 'incorrect_fare')]
    if not fare_discrepancy.empty:
        anomalies_list.append({
            'type': 'Fare Discrepancy',
            'description': f"Transactions where actual fare paid differs from calculated fare (excluding negative balance taps).",
            'count': len(fare_discrepancy),
            'transactions_df': fare_discrepancy
        })
        
    # 3. Negative Balance Tap
    negative_balance_tap = df[df['anomaly_type'] == 'negative_balance_tap']
    if not negative_balance_tap.empty:
        anomalies_list.append({
            'type': 'Negative Balance Tap',
            'description': f"Transactions where tap occurred with insufficient funds.",
            'count': len(negative_balance_tap),
            'transactions_df': negative_balance_tap
        })

    # 4. Over-speed Trips
    over_speed_threshold_kph = 60
    over_speed_trips = df[(df['simulated_speed_kph'] > over_speed_threshold_kph) & (df['anomaly_type'] == 'over_speed_trip')]
    if not over_speed_trips.empty:
        anomalies_list.append({
            'type': 'Over-speed Trip',
            'description': f"Trips with average speed exceeding {over_speed_threshold_kph} km/h (potential data error).",
            'count': len(over_speed_trips),
            'transactions_df': over_speed_trips
        })
    
    # 5. Potential Revenue Loss (Aggregated loss due to specific anomalies)
    revenue_loss_anomalies_df = df[df['anomaly_type'].isin(['missing_tap_out', 'incorrect_fare', 'negative_balance_tap'])]
    if not revenue_loss_anomalies_df.empty: # Check if this df is not empty
        total_lost = (revenue_loss_anomalies_df['calculated_fare'] - revenue_loss_anomalies_df['actual_fare_paid']).sum()
        if total_lost > 0: # Only add if there's actual loss
            anomalies_list.append({
                'type': 'Total Estimated Revenue Loss',
                'description': f"Estimated total revenue loss from specific anomalies.",
                'count': len(revenue_loss_anomalies_df),
                'transactions_df': revenue_loss_anomalies_df,
                'revenue_lost': total_lost
            })

    return anomalies_list

# --- Helper Functions (Defined outside cached functions as they are constants/simple ops) ---
def is_peak_hour(dt_obj, peak_periods):
    if dt_obj.weekday() >= 5: # Saturday or Sunday, assume not peak fare hours for simplicity in fare calculation
        return False
    for start_h, start_m, end_h, end_m in peak_periods:
        start_time = dt_obj.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end_time = dt_obj.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if start_time <= dt_obj <= end_time:
            return True
    return False

def get_fare(distance_km, is_peak_time, fare_structure_peak, fare_structure_offpeak):
    fare_dict = fare_structure_peak if is_peak_time else fare_structure_offpeak
    for (min_dist, max_dist), fare_val in fare_dict.items():
        if min_dist <= distance_km <= max_dist:
            return fare_val
    return 0.0 # Return 0 if no matching fare band (shouldn't happen with np.inf)

# --- FARE AND PEAK HOUR DEFINITIONS (Constants) ---
rea_vaya_fares_peak = {
    (0.01, 5.0): 7.00, (5.01, 10.0): 8.50, (10.01, 15.0): 10.00,
    (15.01, 25.0): 11.50, (25.01, 35.0): 13.00, (35.01, np.inf): 14.50
}
rea_vaya_fares_offpeak = {k: round(v * 0.9, 2) for k, v in rea_vaya_fares_peak.items()}

myciti_peak_hours = [(6, 45, 8, 0), (16, 15, 17, 30)]

# --- Main Dashboard Logic ---

# Sidebar for simulation parameters
st.sidebar.header("Simulation & Filters")

num_transactions = st.sidebar.slider(
    "Number of Simulated Transactions",
    min_value=1000, max_value=50000, value=10000, step=1000
)
simulation_period_days = st.sidebar.slider(
    "Simulation Period (Days)",
    min_value=30, max_value=365, value=90, step=30
)

# Load routes data
active_routes_df = load_and_prepare_routes('Integrated_rapid_transit_(IRT)_system_MyCiTi_Bus_Routes.csv')

# Simulate transactions
current_time = datetime(2025, 6, 18, 10, 50, 17) # Fixed start time for consistent simulation
df_transactions = simulate_transactions(
    num_transactions, active_routes_df, current_time, simulation_period_days,
    myciti_peak_hours, rea_vaya_fares_peak, rea_vaya_fares_offpeak
)

# Only proceed if transactions were successfully simulated
if not df_transactions.empty:
    # Detect anomalies
    detected_anomalies_report = detect_transaction_anomalies(df_transactions)

    # --- KPIs Section ---
    st.header("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    total_transactions = len(df_transactions)
    total_calculated_revenue = df_transactions['calculated_fare'].sum()
    total_actual_revenue = df_transactions['actual_fare_paid'].sum()
    total_anomalies_count = df_transactions[df_transactions['anomaly_type'] != 'none'].shape[0]

    col1.metric("Total Transactions", f"{total_transactions:,}")
    col2.metric("Total Expected Revenue", f"R {total_calculated_revenue:,.2f}")
    col3.metric("Total Actual Revenue", f"R {total_actual_revenue:,.2f}")
    col4.metric("Total Anomalies Detected", f"{total_anomalies_count:,} ({total_anomalies_count/total_transactions:.2%})")

    st.markdown("---")

    # --- Anomaly Insights ---
    st.header("Anomaly Insights")

    # Filter out 'none' for plotting anomaly types
    anomalies_only_df = df_transactions[df_transactions['anomaly_type'] != 'none']

    if not anomalies_only_df.empty:
        anomaly_types_counts = anomalies_only_df['anomaly_type'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=anomaly_types_counts.index, y=anomaly_types_counts.values, ax=ax1, palette='viridis')
        ax1.set_title("Distribution of Detected Anomaly Types")
        ax1.set_xlabel("Anomaly Type")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)
        plt.close(fig1) # Close figure to prevent memory issues
    else:
        st.info("No specific anomalies detected for visualization (all transactions are 'none').")


    st.subheader("Potential Revenue Loss by Anomaly Type")
    # Calculate revenue loss for relevant anomaly types
    revenue_loss_by_anomaly = anomalies_only_df.groupby('anomaly_type').apply(
        lambda x: (x['calculated_fare'] - x['actual_fare_paid']).sum()
    ).sort_values(ascending=False)

    # Filter for types that actually caused a loss (positive loss)
    revenue_loss_by_anomaly = revenue_loss_by_anomaly[revenue_loss_by_anomaly > 0]

    if not revenue_loss_by_anomaly.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=revenue_loss_by_anomaly.index, y=revenue_loss_by_anomaly.values, ax=ax2, palette='magma')
        ax2.set_title("Estimated Revenue Loss (R) by Anomaly Type")
        ax2.set_xlabel("Anomaly Type")
        ax2.set_ylabel("Estimated Revenue Loss (Rands)")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)
        plt.close(fig2) # Close figure
    else:
        st.info("No significant revenue loss detected from specific anomaly types.")

    st.markdown("---")

    # --- Temporal Trends of Transactions and Anomalies ---
    st.header("Temporal Trends")

    col_ts1, col_ts2 = st.columns(2)

    # Daily Transactions
    daily_transactions = df_transactions.groupby('tap_in_date').size().reset_index(name='count')
    if not daily_transactions.empty:
        fig_daily_trans, ax_daily_trans = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tap_in_date', y='count', data=daily_transactions, ax=ax_daily_trans)
        ax_daily_trans.set_title("Daily Transaction Volume")
        ax_daily_trans.set_xlabel("Date")
        ax_daily_trans.set_ylabel("Number of Transactions")
        ax_daily_trans.tick_params(axis='x', rotation=45)
        col_ts1.pyplot(fig_daily_trans)
        plt.close(fig_daily_trans)
    else:
        col_ts1.info("No daily transaction data to display.")

    # Daily Anomalies
    daily_anomalies = anomalies_only_df.groupby('tap_in_date').size().reset_index(name='count')
    if not daily_anomalies.empty:
        fig_daily_anom, ax_daily_anom = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tap_in_date', y='count', data=daily_anomalies, ax=ax_daily_anom, color='red')
        ax_daily_anom.set_title("Daily Anomaly Volume")
        ax_daily_anom.set_xlabel("Date")
        ax_daily_anom.set_ylabel("Number of Anomalies")
        ax_daily_anom.tick_params(axis='x', rotation=45)
        col_ts2.pyplot(fig_daily_anom)
        plt.close(fig_daily_anom)
    else:
        col_ts2.info("No daily anomaly data to display.")


    col_ts3, col_ts4 = st.columns(2)

    # Hourly Transactions
    hourly_transactions = df_transactions.groupby('tap_in_hour').size().reset_index(name='count')
    if not hourly_transactions.empty:
        fig_hourly_trans, ax_hourly_trans = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tap_in_hour', y='count', data=hourly_transactions, ax=ax_hourly_trans)
        ax_hourly_trans.set_title("Hourly Transaction Volume")
        ax_hourly_trans.set_xlabel("Hour of Day")
        ax_hourly_trans.set_ylabel("Number of Transactions")
        col_ts3.pyplot(fig_hourly_trans)
        plt.close(fig_hourly_trans)
    else:
        col_ts3.info("No hourly transaction data to display.")

    # Hourly Anomalies
    hourly_anomalies = anomalies_only_df.groupby('tap_in_hour').size().reset_index(name='count')
    if not hourly_anomalies.empty:
        fig_hourly_anom, ax_hourly_anom = plt.subplots(figsize=(10, 5))
        sns.lineplot(x='tap_in_hour', y='count', data=hourly_anomalies, ax=ax_hourly_anom, color='red')
        ax_hourly_anom.set_title("Hourly Anomaly Volume")
        ax_hourly_anom.set_xlabel("Hour of Day")
        ax_hourly_anom.set_ylabel("Number of Anomalies")
        col_ts4.pyplot(fig_hourly_anom)
        plt.close(fig_hourly_anom)
    else:
        col_ts4.info("No hourly anomaly data to display.")

    st.markdown("---")

    # --- Raw Data View ---
    st.header("Raw Simulated Data")
    st.markdown("Explore a sample of the generated transaction data.")

    # Filters for the raw data table
    filter_col1, filter_col2 = st.columns(2)
    selected_anomaly_type = filter_col1.selectbox(
        "Filter by Anomaly Type",
        options=['All'] + list(df_transactions['anomaly_type'].unique())
    )
    selected_route_number = filter_col2.selectbox(
        "Filter by Route Number",
        options=['All'] + sorted(df_transactions['route_number'].unique().tolist())
    )

    filtered_df = df_transactions.copy()
    if selected_anomaly_type != 'All':
        filtered_df = filtered_df[filtered_df['anomaly_type'] == selected_anomaly_type]
    if selected_route_number != 'All':
        filtered_df = filtered_df[filtered_df['route_number'] == selected_route_number]

    st.dataframe(filtered_df.head(1000)) # Display first 1000 rows of filtered data for performance
    st.info(f"Displaying {len(filtered_df)} rows (max 1000 shown).")

else:
    st.warning("Could not simulate transactions. Please check the route data file and simulation parameters.")


st.markdown("---")
st.caption("Dashboard created using simulated data for demonstration purposes. Fare structures are based on publicly available information for MyCiTi and Rea Vaya (2025 forecasts).")

