import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from datetime import datetime
from fpdf import FPDF
import io
import tempfile

st.set_page_config(page_title="NitroCapt Energy Dashboard", layout="wide")

section = st.sidebar.radio("Navigation", ["Optimization Dashboard", "LCOE Trends", "Carbon Analysis", "Battery ROI", "Carbon Offset", "PPA Simulator", "Battery Trends Analysis"])

st.image("nitrocapt_logo.png", width=150)

if "shared_demand_df" not in st.session_state:
    st.session_state.shared_demand_df = None
if "shared_price_df" not in st.session_state:
    st.session_state.shared_price_df = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None

if section == "Optimization Dashboard":
    st.title("🔋 NitroCapt Energy Optimization Dashboard")

    with st.sidebar:
        demand_file = st.file_uploader("Upload NitroCapt Energy Demand (CSV)", type=["csv"], key="demand")
        price_file = st.file_uploader("Upload Nord Pool Prices (CSV)", type=["csv"], key="price")

        ppa_price = st.number_input("PPA Price (EUR/MWh)", value=50.0)
        st.session_state.ppa_price = ppa_price
        battery_capacity = st.number_input("Battery Capacity (MWh)", value=14.0)
        battery_efficiency = st.number_input("Battery Efficiency (%)", value=90.0)
        battery_dod = st.number_input("Depth of Discharge (%)", value=80.0)
        battery_hours = st.number_input("Storage Duration (Hours)", value=4.0)
        waste_heat_hourly = st.number_input("Waste Heat (MWh/hour)", value=7.5)
        waste_heat_price = st.number_input("Waste Heat Price (EUR/MWh)", value=35.0)

        st.markdown("---")
        st.subheader("📈 PPA Sensitivity Analysis")
        ppa_min = st.number_input("Min PPA (EUR/MWh)", value=30.0)
        ppa_max = st.number_input("Max PPA (EUR/MWh)", value=100.0)
        ppa_step = st.number_input("Step (EUR)", value=10.0)

    if (demand_file and price_file) or (st.session_state.shared_demand_df is not None and st.session_state.shared_price_df is not None):
        if demand_file is not None:
            demand_df = pd.read_csv(demand_file)
            st.session_state.shared_demand_df = demand_df.copy()
        else:
            demand_df = st.session_state.shared_demand_df.copy()

        if price_file is not None:
            price_df = pd.read_csv(price_file)
            st.session_state.shared_price_df = price_df.copy()
        else:
            price_df = st.session_state.shared_price_df.copy()

        demand_df['Timestamp'] = pd.to_datetime(demand_df['Timestamp'])
        price_df['Timestamp'] = pd.to_datetime(price_df['Timestamp'], errors='coerce')
        df = pd.merge(demand_df, price_df, on='Timestamp')

        df['Base_Cost_EUR'] = df['Power_Demand_MW'] * np.minimum(df['Grid_Price_EUR_per_MWh'], ppa_price)
        base_cost = df['Base_Cost_EUR'].sum()

        max_energy = battery_capacity * (battery_dod / 100)
        battery_eff = battery_efficiency / 100
        battery_storage = 0

        charge_list, discharge_list, cost_list, source_list = [], [], [], []

        for _, row in df.iterrows():
            demand = row['Power_Demand_MW']
            grid_price = row['Grid_Price_EUR_per_MWh']
            cost = 0
            charge = discharge = 0

            if grid_price < ppa_price and battery_storage + battery_hours <= max_energy:
                charge = battery_hours
                battery_storage += charge * battery_eff
                cost = demand * grid_price
                source = "Spot"
            elif battery_storage >= battery_hours:
                discharge = battery_hours
                battery_storage -= discharge / battery_eff
                remaining_demand = demand - discharge
                cost = (discharge * 0) + (remaining_demand * ppa_price)
                source = "Battery"
            else:
                cost = demand * ppa_price if grid_price > ppa_price else demand * grid_price
                source = "PPA" if grid_price > ppa_price else "Spot"

            charge_list.append(charge)
            discharge_list.append(discharge)
            cost_list.append(cost)
            source_list.append(source)

        df['Battery_Charge_MW'] = charge_list
        df['Battery_Discharge_MW'] = discharge_list
        df['Final_Cost_EUR'] = cost_list
        df['Source'] = source_list

        final_cost = df['Final_Cost_EUR'].sum()
        savings = base_cost - final_cost
        waste_heat_energy_mwh = waste_heat_hourly * 24 * 365
        waste_heat_total_revenue = waste_heat_energy_mwh * waste_heat_price

        st.metric("Base Cost (No Battery)", f"{base_cost:,.2f} EUR")
        st.metric("Potential Waste Heat Revenue", f"{waste_heat_total_revenue:,.2f} EUR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimized Cost (Battery)", f"{final_cost:,.2f} EUR")
        col2.metric("Savings With Battery", f"{savings:,.2f} EUR")
        col3.metric("Battery Capacity Used", f"{max_energy:.1f} MWh")

        st.subheader("🔍 Energy Source Distribution")
        source_counts = df['Source'].value_counts()
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(source_counts, labels=source_counts.index, autopct="%1.1f%%", textprops={'fontsize': 8})
        st.pyplot(fig)

        st.subheader("📉 Hourly Optimization Snapshot")
        st.dataframe(df, use_container_width=True)

        st.subheader("📊 LCOE vs PPA Price")
        ppa_range = np.arange(ppa_min, ppa_max + 1, ppa_step)
        lcoe_results = []
        for price in ppa_range:
            tmp_costs = df['Power_Demand_MW'] * np.minimum(df['Grid_Price_EUR_per_MWh'], price)
            lcoe = tmp_costs.sum() / df['Power_Demand_MW'].sum()
            lcoe_results.append((price, round(lcoe, 2)))

        lcoe_df = pd.DataFrame(lcoe_results, columns=["PPA_Price", "LCOE"])
        st.line_chart(lcoe_df.set_index("PPA_Price"))
        st.dataframe(lcoe_df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Table as CSV", data=csv, file_name="optimization_results.csv", mime="text/csv")

        class PDF(FPDF):
            def footer(self):
                self.set_y(-10)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()} | www.nitrocapt.com | NitroCapt AB, Uppsala, Sweden", 0, 0, "C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.image("nitrocapt_logo.png", x=10, y=8, w=33)
        pdf.ln(20)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "NitroCapt Energy Optimization Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, f"""
Input Parameters:
- PPA Price: {ppa_price} EUR/MWh
- Battery Capacity: {battery_capacity} MWh
- Battery Efficiency: {battery_efficiency}%
- Depth of Discharge: {battery_dod}%
- Storage Duration: {battery_hours} h
- Waste Heat: {waste_heat_hourly} MWh/h at {waste_heat_price} EUR/MWh

Results:
- Base Cost (No Battery): {base_cost:,.2f} EUR
- Final Cost (Battery): {final_cost:,.2f} EUR
- Savings: {savings:,.2f} EUR
- Waste Heat Revenue: {waste_heat_total_revenue:,.2f} EUR
""")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_pie:
            fig.savefig(tmp_pie.name, format='png')
            pdf.image(tmp_pie.name, x=10, y=None, w=100)

        lcoe_fig, ax2 = plt.subplots()
        ax2.plot(lcoe_df['PPA_Price'], lcoe_df['LCOE'], marker='o')
        ax2.set_title("LCOE vs PPA Price")
        ax2.set_xlabel("PPA Price (EUR/MWh)")
        ax2.set_ylabel("LCOE (EUR/MWh)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_lcoe:
            lcoe_fig.savefig(tmp_lcoe.name, format='png')
            pdf.image(tmp_lcoe.name, x=10, y=None, w=100)

        pdf_output = pdf.output(dest='S').encode('latin-1', 'ignore')
        b64_pdf = base64.b64encode(pdf_output).decode('utf-8')
        st.markdown(f'<a href="data:application/pdf;base64,{b64_pdf}" download="NitroCapt_Optimization_Report.pdf">📄 Download PDF Report</a>', unsafe_allow_html=True)

        st.session_state.last_results = df.copy()

elif section == "LCOE Trends":
    st.title("📊 LCOE Trends Analysis")

    if st.session_state.shared_demand_df is not None and st.session_state.shared_price_df is not None:
        demand_df = st.session_state.shared_demand_df.copy()
        price_df = st.session_state.shared_price_df.copy()
        df = pd.merge(demand_df, price_df, on='Timestamp')

        battery_range = np.arange(14, 41, 2)
        battery_eff = 0.9
        battery_dod = 0.8
        storage_hours = 4

        results = []

        for cap in battery_range:
            max_energy = cap * battery_dod
            battery_storage = 0
            total_cost = 0

            for _, row in df.iterrows():
                demand = row['Power_Demand_MW']
                grid_price = row['Grid_Price_EUR_per_MWh']

                if grid_price < 50 and battery_storage + storage_hours <= max_energy:
                    battery_storage += storage_hours * battery_eff
                    total_cost += demand * grid_price
                elif battery_storage >= storage_hours:
                    battery_storage -= storage_hours / battery_eff
                    total_cost += (demand - storage_hours) * 50
                else:
                    total_cost += demand * 50

            lcoe = total_cost / df['Power_Demand_MW'].sum()
            results.append((cap, round(lcoe, 2)))

        lcoe_df = pd.DataFrame(results, columns=["Battery_Capacity_MWh", "LCOE"])
        st.subheader("🔋 LCOE vs Battery Capacity (14–40 MWh)")
        st.line_chart(lcoe_df.set_index("Battery_Capacity_MWh"))
        st.dataframe(lcoe_df, use_container_width=True)
    else:
        st.warning("⚠️ Please upload data in the Optimization Dashboard first.")

elif section == "Carbon Analysis":
    st.title("🌱 Carbon Emissions Analysis")

    if (
        st.session_state.get("shared_demand_df") is not None and
        st.session_state.get("shared_price_df") is not None and
        st.session_state.get("last_results") is not None and
        st.session_state.get("ppa_price") is not None
    ):
        st.sidebar.subheader("Carbon Intensity Inputs")
        use_live_data = st.sidebar.checkbox("Use Real-Time Data for Sweden", value=False)

        if use_live_data:
            import requests
            api_key = "pwOaRfVQBwSCw1PsVkFl"  # Replace with your real API key
            url = "https://api.electricitymap.org/v3/carbon-intensity/latest?zone=SE"
            headers = {"auth-token": api_key}
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                try:
                    json_response = response.json()
                    if "carbonIntensity" in json_response:
                        grid_intensity = json_response["carbonIntensity"]
                        st.sidebar.metric("Real-Time Grid Intensity (Sweden)", f"{grid_intensity} gCO₂/kWh")
                    else:
                        st.sidebar.error("Unexpected response structure.")
                        grid_intensity = st.sidebar.number_input("Grid Carbon Intensity (gCO₂/kWh)", value=350.0)
                except Exception as e:
                    st.sidebar.error("Unexpected response structure.")
                    grid_intensity = st.sidebar.number_input("Grid Carbon Intensity (gCO₂/kWh)", value=350.0)
            else:
                st.sidebar.error("Failed to retrieve real-time data.")
                grid_intensity = st.sidebar.number_input("Grid Carbon Intensity (gCO₂/kWh)", value=350.0)
        else:
            grid_intensity = st.sidebar.number_input("Grid Carbon Intensity (gCO₂/kWh)", value=350.0)

        ppa_intensity = st.sidebar.number_input("PPA Carbon Intensity (gCO₂/kWh)", value=0.0)

        demand_df = st.session_state.shared_demand_df.copy()
        price_df = st.session_state.shared_price_df.copy()
        df = pd.merge(demand_df, price_df, on='Timestamp')
        optimized_df = st.session_state.last_results.copy()

        ppa_price = st.session_state.ppa_price

        total_energy = df['Power_Demand_MW'].sum()

        df['Grid_vs_PPA_Emissions'] = df.apply(
            lambda row: row['Power_Demand_MW'] * (grid_intensity if row['Grid_Price_EUR_per_MWh'] < ppa_price else ppa_intensity), axis=1)
        emissions_without_battery = df['Grid_vs_PPA_Emissions'].sum() / 1000

        required_cols = ['Battery_Charge_MW', 'Battery_Discharge_MW', 'Power_Demand_MW', 'Source']
        if all(col in optimized_df.columns for col in required_cols):
            optimized_df['Battery_Charge_Emissions'] = optimized_df['Battery_Charge_MW'] * grid_intensity
            optimized_df['Battery_Discharge_Emissions'] = 0

            optimized_df['Battery_vs_Grid_Emissions'] = optimized_df.apply(
                lambda row: row['Battery_Discharge_Emissions'] +
                            (row['Power_Demand_MW'] - row['Battery_Discharge_MW']) *
                            (ppa_intensity if row['Source'] == "PPA" else grid_intensity), axis=1)

            emissions_with_battery = (
                optimized_df['Battery_Charge_Emissions'].sum() +
                optimized_df['Battery_vs_Grid_Emissions'].sum()
            ) / 1000

            carbon_savings = emissions_without_battery - emissions_with_battery

            st.metric("Emissions Without Battery", f"{emissions_without_battery:,.2f} tons CO₂")
            st.metric("Emissions With Battery", f"{emissions_with_battery:,.2f} tons CO₂")
            st.metric("Carbon Saved", f"{carbon_savings:,.2f} tons CO₂")

            st.subheader("📊 Emissions Breakdown")
            emissions_data = pd.DataFrame({
                "Scenario": ["Without Battery", "With Battery"],
                "Emissions (tons CO₂)": [emissions_without_battery, emissions_with_battery]
            })
            st.bar_chart(emissions_data.set_index("Scenario"))

            st.session_state['emissions_summary'] = {
                "emissions_without_battery": emissions_without_battery,
                "emissions_with_battery": emissions_with_battery,
                "carbon_savings": carbon_savings
            }

            # Include in report
            if 'pdf' in st.session_state:
                import tempfile
                import matplotlib.pyplot as plt
                pdf = st.session_state['pdf']
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, "Carbon Emissions Analysis", ln=True)
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 8, f"""
Carbon Emissions:
- Emissions Without Battery: {emissions_without_battery:,.2f} tons CO2
- Emissions With Battery: {emissions_with_battery:,.2f} tons CO2
- Carbon Saved: {carbon_savings:,.2f} tons CO2
                """)

                fig, ax = plt.subplots()
                emissions_data.set_index("Scenario").plot(kind='bar', ax=ax, legend=False)
                ax.set_ylabel("Emissions (tons CO2)")
                ax.set_title("Emissions Breakdown")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_emission_chart:
                    fig.savefig(tmp_emission_chart.name, format='png')
                    pdf.image(tmp_emission_chart.name, x=10, y=None, w=100)

        else:
            st.warning("⚠️ The required columns are missing from the optimized results. Please rerun the optimization with battery enabled.")
    else:
        st.warning("⚠️ Please upload data and run optimization with battery enabled before viewing carbon analysis.")


if section == "Battery ROI":
    st.title("⮍ Battery ROI Analysis")

    st.sidebar.subheader("Battery Configuration")

    # Independent battery capacity input for Battery ROI section
    battery_capacity_roi = st.sidebar.number_input("Battery Capacity (MWh)", value=14.0, key="battery_capacity_roi")

    st.sidebar.subheader("Cost Inputs (EUR/kWh)")
    cell_cost = st.sidebar.number_input("Battery Cells Cost", value=350.0, key="cell_cost")
    electronics_cost = st.sidebar.number_input("Power Electronics (Inverters, Converters)", value=60.0, key="electronics_cost")
    bos_cost = st.sidebar.number_input("Balance of System (BOS)", value=35.0, key="bos_cost")
    epc_cost_per_kwh = st.sidebar.number_input("EPC & Construction", value=40.0, key="epc_cost")
    misc_cost = st.sidebar.number_input("Miscellaneous (Permits, Grid Connection, Project Mgmt)", value=50.0, key="misc_cost")

    # CAPEX Calculations
    battery_capacity_kwh = battery_capacity_roi * 1000
    capex_per_kwh_total = cell_cost + electronics_cost + bos_cost + epc_cost_per_kwh + misc_cost
    total_capex = battery_capacity_kwh * capex_per_kwh_total

    st.markdown("---")
    st.subheader("💰 CAPEX Summary")
    st.metric("CAPEX (EUR/kWh)", f"{capex_per_kwh_total:,.2f}")
    st.metric("Total CAPEX (EUR)", f"{total_capex:,.2f}")

    st.sidebar.subheader("Operational Inputs")
    opex_percent = st.sidebar.number_input("OPEX (% of CAPEX)", value=2.0, step=0.5, key="opex_percent")
    battery_lifetime = st.sidebar.number_input("Battery Lifetime (years)", value=30, step=1, key="battery_lifetime")
    degradation_rate = st.sidebar.number_input("Battery Degradation (% per year)", value=2.0, step=0.1, key="degradation_rate")

    opex_annual = (opex_percent / 100) * total_capex
    total_opex = opex_annual * battery_lifetime
    total_system_cost = total_capex + total_opex

    # Pull optimization input data
    savings_with_battery = None
    if st.session_state.get("last_results") is not None and st.session_state.get("ppa_price") is not None:
        demand_df = st.session_state.shared_demand_df.copy()
        price_df = st.session_state.shared_price_df.copy()
        df = pd.merge(demand_df, price_df, on='Timestamp')

        ppa_price = st.session_state.ppa_price
        max_energy = battery_capacity_roi * 0.8  # DoD fixed at 80%
        battery_eff = 0.9  # Efficiency fixed at 90%
        battery_hours = 4  # Same as optimization default
        battery_storage = 0

        final_cost_list = []

        for _, row in df.iterrows():
            demand = row['Power_Demand_MW']
            grid_price = row['Grid_Price_EUR_per_MWh']
            cost = 0

            if grid_price < ppa_price and battery_storage + battery_hours <= max_energy:
                battery_storage += battery_hours * battery_eff
                cost = demand * grid_price
            elif battery_storage >= battery_hours:
                battery_storage -= battery_hours / battery_eff
                cost = (demand - battery_hours) * ppa_price
            else:
                cost = demand * (ppa_price if grid_price > ppa_price else grid_price)

            final_cost_list.append(cost)

        df['Final_Cost_EUR'] = final_cost_list
        base_cost = (df['Power_Demand_MW'] * np.minimum(df['Grid_Price_EUR_per_MWh'], ppa_price)).sum()
        final_cost = df['Final_Cost_EUR'].sum()
        yearly_savings = base_cost - final_cost

        # Apply degradation
        rate = degradation_rate / 100
        discounted_savings = sum([(1 - rate) ** year for year in range(battery_lifetime)]) * yearly_savings
        savings_with_battery = discounted_savings

    st.subheader("📊 Cost Breakdown")
    cost_data = {
        "Component": [
            "Battery Cells", "Power Electronics", "BOS", "EPC & Construction",
            "Miscellaneous (incl. Grid Connection)", "Total OPEX ({} yrs)".format(battery_lifetime), "Total System Cost"
        ],
        "Cost (EUR)": [
            battery_capacity_kwh * cell_cost,
            battery_capacity_kwh * electronics_cost,
            battery_capacity_kwh * bos_cost,
            battery_capacity_kwh * epc_cost_per_kwh,
            battery_capacity_kwh * misc_cost,
            total_opex,
            total_system_cost
        ]
    }
    cost_df = pd.DataFrame(cost_data)
    st.dataframe(cost_df, use_container_width=True)

    if savings_with_battery is not None:
        roi = ((savings_with_battery - total_system_cost) / total_system_cost) * 100
        payback = total_system_cost / (savings_with_battery / battery_lifetime) if savings_with_battery > 0 else None

                # Plot degradation impact
        st.subheader("📉 Battery Degradation Impact on Annual Savings")
        degradation_years = list(range(1, battery_lifetime + 1))
        savings_by_year = [yearly_savings * ((1 - rate) ** (year - 1)) for year in degradation_years]
        savings_df = pd.DataFrame({
            "Year": degradation_years,
            "Annual Savings (EUR)": savings_by_year
        }).set_index("Year")
        st.line_chart(savings_df)

        st.subheader("📈 ROI Summary")
        st.metric("Estimated Savings Over Lifetime (EUR)", f"{savings_with_battery:,.2f}")
        st.metric("Total System Cost (EUR)", f"{total_system_cost:,.2f}")
        st.metric("ROI (%)", f"{roi:,.2f}%")
        if payback:
            st.metric("Payback Period (years)", f"{payback:.2f}")
    else:
        st.warning("⚠️ Please run optimization first to estimate savings.")

elif section == "Carbon Offset":
    st.title("🌍 Carbon Offset Dashboard")

    with st.sidebar:
        st.subheader("📉 Carbon Offset Inputs")

        carbon_savings = abs(st.session_state.get("emissions_summary", {}).get("carbon_savings", 0.0))
        waste_heat_energy_mwh = st.number_input("Waste Heat Energy (MWh/year)", value=7.5 * 24 * 365, step=10.0)
        waste_heat_carbon_offset = st.number_input("Waste Heat Carbon Offset (kg CO₂/MWh)", value=200.0, step=10.0)

        st.markdown("---")
        st.subheader("🌍 Select Carbon Market")
        market_option = st.selectbox("Choose Market:", ["EU ETS (65 €/ton)", "Voluntary (10 €/ton)", "US CCA (35 €/ton)"])

        if "EU ETS" in market_option:
            carbon_credit_price = 65.0
        elif "Voluntary" in market_option:
            carbon_credit_price = 10.0
        elif "US CCA" in market_option:
            carbon_credit_price = 35.0
        else:
            carbon_credit_price = 50.0

        manual_price = st.checkbox("🔧 Manually adjust credit price")
        if manual_price:
            carbon_credit_price = st.number_input("Manual Carbon Credit Price (EUR/ton CO₂)", value=carbon_credit_price, step=1.0)

        st.markdown("---")
        st.subheader("🧾 Regulatory Compliance")
        target_reduction = st.number_input("Target Emissions Reduction (tons CO₂/year)", value=100.0, step=10.0)

    # Calculate total offset
    waste_heat_co2_tons = (waste_heat_energy_mwh * waste_heat_carbon_offset) / 1000
    total_offset_tons = carbon_savings + waste_heat_co2_tons
    revenue = total_offset_tons * carbon_credit_price

    st.subheader("🌱 Offset Summary")
    st.markdown(
        f"""
        <div style='display: flex; gap: 6rem; font-size: 20px;'>
            <div><strong>Battery Carbon Savings</strong><br>{carbon_savings:,.2f}</div>
            <div><strong>Waste Heat Offset</strong><br>{waste_heat_co2_tons:,.2f}</div>
            <div><strong>Total Offset</strong><br>{total_offset_tons:,.2f}</div>
            <div><strong>Potential Revenue</strong><br>{revenue:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Offset Breakdown")
    offset_data = pd.DataFrame({
        "Source": ["Battery Savings", "Waste Heat"],
        "Offset (tons CO₂)": [carbon_savings, waste_heat_co2_tons]
    })
    st.bar_chart(offset_data.set_index("Source"))

    # 📏 Compliance Comparison
    st.subheader("📋 Compliance Status")
    percent_achieved = (total_offset_tons / target_reduction) * 100 if target_reduction > 0 else 0
    if total_offset_tons >= target_reduction:
        st.success(f"✅ On Track: Achieved {percent_achieved:.1f}% of target")
    else:
        st.warning(f"⚠️ Below Target: Achieved {percent_achieved:.1f}% of target")

    st.progress(min(1.0, percent_achieved / 100))

elif section == "Carbon Offset":
    st.title("🌍 Carbon Offset Dashboard")

    with st.sidebar:
        st.subheader("📉 Carbon Offset Inputs")

        carbon_savings = abs(st.session_state.get("emissions_summary", {}).get("carbon_savings", 0.0))
        waste_heat_energy_mwh = st.number_input("Waste Heat Energy (MWh/year)", value=7.5 * 24 * 365, step=10.0)
        waste_heat_carbon_offset = st.number_input("Waste Heat Carbon Offset (kg CO₂/MWh)", value=200.0, step=10.0)

        st.markdown("---")
        st.subheader("🌍 Select Carbon Market")
        market_option = st.selectbox("Choose Market:", ["EU ETS (65 €/ton)", "Voluntary (10 €/ton)", "US CCA (35 €/ton)"])

        if "EU ETS" in market_option:
            carbon_credit_price = 65.0
        elif "Voluntary" in market_option:
            carbon_credit_price = 10.0
        elif "US CCA" in market_option:
            carbon_credit_price = 35.0
        else:
            carbon_credit_price = 50.0

        manual_price = st.checkbox("🔧 Manually adjust credit price")
        if manual_price:
            carbon_credit_price = st.number_input("Manual Carbon Credit Price (EUR/ton CO₂)", value=carbon_credit_price, step=1.0)

        st.markdown("---")
        st.subheader("🧾 Regulatory Compliance")
        target_reduction = st.number_input("Target Emissions Reduction (tons CO₂/year)", value=100.0, step=10.0)

    # Calculate total offset
    waste_heat_co2_tons = (waste_heat_energy_mwh * waste_heat_carbon_offset) / 1000
    total_offset_tons = carbon_savings + waste_heat_co2_tons
    revenue = total_offset_tons * carbon_credit_price

    st.subheader("🌱 Offset Summary")
    st.markdown(
        f"""
        <div style='display: flex; justify-content: space-between; font-size: 20px;'>
            <div><strong>Battery Carbon Savings</strong><br>{carbon_savings:,.2f}</div>
            <div><strong>Waste Heat Offset</strong><br>{waste_heat_co2_tons:,.2f}</div>
            <div><strong>Total Offset</strong><br>{total_offset_tons:,.2f}</div>
            <div><strong>Potential Revenue</strong><br>{revenue:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📊 Offset Breakdown")
    offset_data = pd.DataFrame({
        "Source": ["Battery Savings", "Waste Heat"],
        "Offset (tons CO₂)": [carbon_savings, waste_heat_co2_tons]
    })
    st.bar_chart(offset_data.set_index("Source"))

    # 📏 Compliance Comparison
    st.subheader("📋 Compliance Status")
    percent_achieved = (total_offset_tons / target_reduction) * 100 if target_reduction > 0 else 0
    if total_offset_tons >= target_reduction:
        st.success(f"✅ On Track: Achieved {percent_achieved:.1f}% of target")
    else:
        st.warning(f"⚠️ Below Target: Achieved {percent_achieved:.1f}% of target")

    st.progress(min(1.0, percent_achieved / 100))

elif section == "PPA Simulator":
    st.title("📑 PPA Contract Simulator")

    with st.sidebar:
        st.subheader("⚙️ PPA Configuration")
        ppa_type = st.selectbox("Select PPA Type", ["Fixed", "Variable", "Hybrid"])
        fixed_ppa_price = st.number_input("Fixed PPA Price (EUR/MWh)", value=55.0)

        variable_prices = {
            'Jan': 52, 'Feb': 54, 'Mar': 50, 'Apr': 51,
            'May': 53, 'Jun': 55, 'Jul': 57, 'Aug': 56,
            'Sep': 54, 'Oct': 52, 'Nov': 51, 'Dec': 50
        }

        uploaded = st.file_uploader("Upload Monthly Variable Prices (CSV)", type=["csv"])
        if uploaded:
            variable_df = pd.read_csv(uploaded)
            if 'Month' in variable_df.columns and 'Price' in variable_df.columns:
                variable_prices = dict(zip(variable_df['Month'], variable_df['Price']))

    if st.session_state.shared_demand_df is not None and st.session_state.shared_price_df is not None:
        # Use real data
        demand_df = st.session_state.shared_demand_df.copy()
        price_df = st.session_state.shared_price_df.copy()
        df = pd.merge(demand_df, price_df, on="Timestamp")

        df = pd.merge(demand_df, price_df, on="Timestamp")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Ensures proper datetime format
        df['Month'] = df['Timestamp'].dt.strftime('%b')  # Jan, Feb, etc.
        monthly_demand = df.groupby('Month')['Power_Demand_MW'].sum()
        monthly_spot = df.groupby('Month')['Grid_Price_EUR_per_MWh'].mean()

        months = list(variable_prices.keys())
        demand = np.array([monthly_demand.get(m, 0) for m in months])
        spot_prices = np.array([monthly_spot.get(m, 0) for m in months])
        variable_ppa = np.array([variable_prices.get(m, fixed_ppa_price) for m in months])

        fixed_costs = demand * fixed_ppa_price
        variable_costs = demand * variable_ppa
        hybrid_costs = [min(s, fixed_ppa_price) * d for s, d in zip(spot_prices, demand)]

        result_df = pd.DataFrame({
            "Month": months,
            "Demand (MWh)": demand,
            "Spot Price": spot_prices,
            "Variable PPA": variable_ppa,
            "Fixed PPA Cost": fixed_costs,
            "Variable PPA Cost": variable_costs,
            "Hybrid Cost": hybrid_costs
        })

        st.subheader("📊 Monthly Cost Comparison")
        st.dataframe(result_df)

        st.subheader("💡 Annual Cost Comparison")
        total_costs = pd.DataFrame({
            "Strategy": ["Fixed", "Variable", "Hybrid"],
            "Total Cost (EUR)": [fixed_costs.sum(), variable_costs.sum(), sum(hybrid_costs)]
        })
        st.bar_chart(total_costs.set_index("Strategy"))

        best_option = total_costs.loc[total_costs["Total Cost (EUR)"].idxmin(), "Strategy"]
        st.success(f"✅ Recommended Strategy: {best_option}")
    else:
        st.warning("⚠️ Please upload demand and price files in the Optimization Dashboard first.")

elif section == "Battery Trends": st.title("Battery Trends Analysis")
if st.session_state.get("last_results") is not None:
    df = st.session_state.last_results.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Date'] = df['Timestamp'].dt.date

    selected_date = st.sidebar.date_input("Select Date", df['Date'].iloc[0])
    daily_df = df[df['Date'] == selected_date]

    if not daily_df.empty:
        # Calculate battery storage level
        storage_level = 0
        storage_levels = []
        battery_efficiency = 90.0  # Default efficiency, adjust as needed
        max_energy = 14.0 * (80 / 100)  # Default battery capacity and DoD
        efficiency = battery_efficiency / 100

        for _, row in daily_df.iterrows():
            storage_level += row['Battery_Charge_MW'] * efficiency
            storage_level -= row['Battery_Discharge_MW'] / efficiency
            storage_level = max(0, min(storage_level, max_energy))
            storage_levels.append(storage_level)

        daily_df['Storage_Level_MWh'] = storage_levels

        st.subheader(f"🔋 Hourly Battery Behavior on {selected_date}")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(daily_df['Timestamp'], daily_df['Battery_Charge_MW'], label='Charge (MW)', color='green')
        ax.plot(daily_df['Timestamp'], daily_df['Battery_Discharge_MW'], label='Discharge (MW)', color='red')
        ax.plot(daily_df['Timestamp'], daily_df['Power_Demand_MW'], label='Demand (MW)', color='blue', alpha=0.5)
        ax.plot(daily_df['Timestamp'], daily_df['Storage_Level_MWh'], label='Storage Level (MWh)', color='purple', linestyle='--')
        ax.set_xlabel("Time")
        ax.set_ylabel("Power / Storage")
        ax.set_title("Battery Charge, Discharge & Storage Profile")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.subheader("🔍 Data Table")
        st.dataframe(daily_df[['Timestamp', 'Power_Demand_MW', 'Battery_Charge_MW', 'Battery_Discharge_MW', 'Storage_Level_MWh', 'Grid_Price_EUR_per_MWh']], use_container_width=True)

    else:
        st.warning("No data available for the selected date.")
else:
    st.warning("⚠️ Please run the optimization in the Optimization Dashboard first.")