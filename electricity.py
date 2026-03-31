import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import base64
import datetime
import warnings

# --- Page Config ---
st.set_page_config(page_title="Electricity Load Optimizer", layout="wide")
st.markdown("<h1 style='text-align: center;'>⚡ Electricity Load Optimizer</h1><hr>", unsafe_allow_html=True)

# --- Sidebar for API Key (Simple and Works) ---
with st.sidebar:
    st.markdown("## 🔑 AI Configuration")
    st.markdown("Enter your Groq API key to enable AI-powered insights")
    api_key = st.text_input(
        "Groq API Key:", 
        type="password", 
        placeholder="gsk_...",
        help="Get your free API key from https://console.groq.com"
    )
    
    if api_key:
        try:
            client = Groq(api_key=api_key)
            st.success("✅ AI client ready! AI insights are enabled.")
        except Exception as e:
            st.error(f"❌ Invalid API key: {str(e)}")
            client = None
    else:
        client = None
        st.info("ℹ️ Enter your API key to enable AI insights")
    
    st.markdown("---")
    st.markdown("**Need an API key?**")
    st.markdown("[Get free Groq API key](https://console.groq.com)")

# --- Function to auto-detect and rename columns ---
def detect_and_rename_columns(df):
    """Automatically detect and rename all columns to standard names"""
    df_copy = df.copy()
    column_mapping = {}
    used_names = set()
    
    for col in df_copy.columns:
        col_original = col
        col_lower = str(col).lower().strip()
        col_clean = col_lower.split('(')[0].strip()
        
        new_name = None
        
        # Check for datetime columns
        if col_clean in ['timestamp', 'datetime', 'date', 'time']:
            new_name = 'datetime'
        
        # Check for load/demand columns
        elif col_clean in ['totaldemand', 'load', 'demand']:
            new_name = 'load'
        
        # Check for load forecast
        elif col_clean in ['loadforecast', 'forecast']:
            new_name = 'load_forecast'
        
        # Check for wind generation
        elif col_clean in ['windgeneration', 'wind generation']:
            new_name = 'wind_generation'
        
        # Check for hydro generation
        elif col_clean in ['hydrogeneration', 'hydro generation']:
            new_name = 'hydro_generation'
        
        # Check for solar generation
        elif col_clean in ['solargeneration', 'solar generation']:
            new_name = 'solar_generation'
        
        # Check for solar irradiance
        elif col_clean in ['solarirradiance', 'solar irradiance']:
            new_name = 'solar_irradiance'
        
        # Check for temperature
        elif col_clean in ['temperature', 'temp']:
            new_name = 'temperature'
        
        # Check for humidity
        elif col_clean == 'humidity':
            new_name = 'humidity'
        
        # Check for wind speed
        elif col_clean in ['windspeed', 'wind speed']:
            new_name = 'wind_speed'
        
        # Check for price
        elif col_clean == 'price' and 'lag' not in col_lower and 'previous' not in col_lower:
            new_name = 'price'
        
        # Check for coal generation
        elif col_clean in ['coalgeneration', 'coal generation']:
            new_name = 'coal_generation'
        
        # Check for gas generation
        elif col_clean in ['gasgeneration', 'gas generation']:
            new_name = 'gas_generation'
        
        # Check for renewable share
        elif col_clean in ['renewableshare', 'renewable share']:
            new_name = 'renewable_share'
        
        # Check for rolling mean
        elif 'rollingmean' in col_clean:
            new_name = 'rolling_mean_3h'
        
        # Check for rolling std
        elif 'rollingstd' in col_clean:
            new_name = 'rolling_std_3h'
        
        # Check for day of week
        elif col_clean in ['dayofweek', 'day of week']:
            new_name = 'day_of_week'
        
        # Check for hour of day
        elif col_clean in ['hourofday', 'hour of day']:
            new_name = 'hour_of_day'
        
        # Check for holiday flag
        elif col_clean in ['holidayflag', 'holiday flag']:
            new_name = 'holiday_flag'
        
        # For lag columns, keep original names
        elif 'previousprice' in col_clean or 'lag' in col_lower:
            new_name = col_original
        
        # If no mapping found, keep original name
        else:
            new_name = col_original
        
        # Handle duplicates
        if new_name and new_name in used_names:
            base_name = new_name
            counter = 2
            while f"{base_name}_{counter}" in used_names:
                counter += 1
            new_name = f"{base_name}_{counter}"
        
        if new_name:
            column_mapping[col] = new_name
            used_names.add(new_name)
    
    # Apply mapping
    for old_name, new_name in column_mapping.items():
        if old_name in df_copy.columns:
            df_copy.rename(columns={old_name: new_name}, inplace=True)
    
    return df_copy, column_mapping

# --- LLM Analysis Function ---
def llm_analysis(df, peak_load, avg_load, load_factor, renewable_percentage, client):
    if not client:
        return """
        ### ⚠️ AI Analysis Unavailable
        
        **To enable AI-powered insights:**
        
        1. Get your free Groq API key from [console.groq.com](https://console.groq.com)
        2. Enter it in the sidebar on the left
        3. The AI analysis will appear here automatically
        
        **Without AI, you can still:**
        - ✅ View load patterns in the Analysis tab
        - ✅ Run optimization simulations in the Optimization tab
        - ✅ Download CSV reports
        """
    
    data_summary = f"""
    Time Range: {df.index.min()} to {df.index.max()}
    Total Records: {len(df)}
    Peak Load: {peak_load:.2f} MW
    Average Load: {avg_load:.2f} MW
    Load Factor: {load_factor:.2%}
    Renewable Contribution: {renewable_percentage:.2%}
    """
    
    if 'wind_generation' in df.columns:
        data_summary += f"\nWind Generation Avg: {df['wind_generation'].mean():.2f} MW"
    if 'hydro_generation' in df.columns:
        data_summary += f"\nHydro Generation Avg: {df['hydro_generation'].mean():.2f} MW"
    if 'solar_generation' in df.columns:
        data_summary += f"\nSolar Generation Avg: {df['solar_generation'].mean():.2f} MW"
    if 'temperature' in df.columns:
        data_summary += f"\nTemperature Avg: {df['temperature'].mean():.1f} °C"
    
    prompt = f"""
    You are an AI expert in electricity load optimization with renewable energy integration.
    
    DATA SUMMARY:
    {data_summary}
    
    Provide comprehensive analysis in these 4 sections:
    
    1. LOAD PATTERN ANALYSIS:
    - Identify peak demand periods and patterns
    - Analyze load variations throughout the day/week
    - Identify correlations with weather and renewable generation
    
    2. OPTIMIZATION PREDICTIONS:
    - Forecast optimal load distribution for next 7 days
    - Recommend load shifting strategies
    - Suggest renewable energy integration opportunities
    
    3. INTERVENTION IMPACTS:
    - Suggest specific demand-side management interventions
    - Estimate potential energy savings
    - Calculate CO2 reduction potential
    
    4. POLICY RECOMMENDATIONS:
    - Provide actionable policy recommendations
    - Suggest peak load management strategies
    - Recommend renewable energy targets
    
    Format with clear headings and bullet points. Be specific with numbers.
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"""
        ### ❌ AI Analysis Error
        
        **Error Details:** {str(e)}
        
        **Troubleshooting:**
        - Check if your API key is correct (should start with 'gsk_')
        - Ensure you have credits in your Groq account
        - Try regenerating your API key at [console.groq.com](https://console.groq.com)
        """

# --- File Upload ---
uploaded_file = st.file_uploader("📁 C:/Users/kokki/Desktop/ppt/Electricity_Market_Renewable_Generation.csv", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        warnings.filterwarnings('ignore')
        
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.info(f"📊 Found {len(df.columns)} columns")
        
        # Auto-detect and rename columns
        df, renamed_columns = detect_and_rename_columns(df)
        
        mapped_cols = {old: new for old, new in renamed_columns.items() if old != new}
        if mapped_cols:
            st.success(f"✅ Auto-mapped {len(mapped_cols)} columns")
        
        # Check for required columns
        if 'datetime' not in df.columns:
            st.warning("No datetime column found. Creating index from row numbers.")
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
        else:
            with st.spinner("Parsing datetime column..."):
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                if df['datetime'].isnull().any():
                    st.warning("Some datetime values could not be parsed. Using fallback method...")
                    df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')
                st.success(f"✅ Datetime parsed successfully")
        
        if 'load' not in df.columns:
            st.error("No load/demand column found.")
            st.stop()
        
        # Set datetime index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        df = df[df.index.notnull()]
        
        # Calculate metrics
        peak_load = df['load'].max()
        avg_load = df['load'].mean()
        load_factor = avg_load / peak_load if peak_load > 0 else 0
        
        # Calculate renewable percentage
        renewable_cols = []
        for col in ['wind_generation', 'hydro_generation', 'solar_generation']:
            if col in df.columns:
                renewable_cols.append(col)
        
        if renewable_cols:
            total_renewable = df[renewable_cols].sum(axis=1)
            renewable_percentage = (total_renewable.sum() / df['load'].sum()) * 100
        else:
            renewable_percentage = 0
        
        # Tabs UI
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Input Data", "📈 Analysis", "🎯 Optimization", "💡 AI Insights", "📄 Report"])
        
        with tab1:
            st.subheader("Dataset Overview")
            st.dataframe(df.head(20), width='stretch')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Peak Load", f"{peak_load:.2f} MW")
            with col2: st.metric("Average Load", f"{avg_load:.2f} MW")
            with col3: st.metric("Load Factor", f"{load_factor:.2%}")
            with col4: st.metric("Renewable Share", f"{renewable_percentage:.1f}%" if renewable_percentage > 0 else "N/A")
            
            st.line_chart(df['load'], width='stretch')
        
        with tab2:
            st.subheader("Load Pattern Analysis")
            
            # Hourly pattern
            df['hour'] = df.index.hour
            hourly_load = df.groupby('hour')['load'].mean()
            fig_hourly = px.line(x=hourly_load.index, y=hourly_load.values, 
                                 title="Average Load by Hour",
                                 labels={'x': 'Hour', 'y': 'Load (MW)'})
            st.plotly_chart(fig_hourly, width='stretch')
            
            col1, col2 = st.columns(2)
            
            with col1:
                df['day'] = df.index.dayofweek
                daily_load = df.groupby('day')['load'].mean()
                fig_daily = px.bar(x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 
                                   y=daily_load.values,
                                   title="Average Load by Day of Week")
                st.plotly_chart(fig_daily, width='stretch')
            
            with col2:
                load_categories = pd.cut(df['load'], bins=3, labels=['Low', 'Medium', 'High'])
                pie_data = load_categories.value_counts()
                fig_pie = px.pie(values=pie_data.values, names=pie_data.index, title="Load Distribution")
                st.plotly_chart(fig_pie, width='stretch')
            
            # Weather correlation
            weather_cols = ['temperature', 'humidity', 'wind_speed']
            available_weather = [col for col in weather_cols if col in df.columns]
            if available_weather:
                st.subheader("Weather Impact Analysis")
                corr_data = df[['load'] + available_weather].corr()
                fig_corr = px.imshow(corr_data, text_auto=True, title="Correlation: Load vs Weather")
                st.plotly_chart(fig_corr, width='stretch')
            
            # Renewable generation
            if renewable_cols:
                st.subheader("Renewable Generation Analysis")
                renewable_data = df[renewable_cols].mean()
                fig_renewable = px.bar(x=renewable_data.index, y=renewable_data.values,
                                       title="Average Renewable Generation")
                st.plotly_chart(fig_renewable, width='stretch')
                
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=df.index, y=df['load'], name='Load', line=dict(color='blue')))
                for col in renewable_cols:
                    fig_ts.add_trace(go.Scatter(x=df.index, y=df[col], name=col.replace('_', ' ').title()))
                fig_ts.update_layout(title="Load vs Renewable Generation Over Time")
                st.plotly_chart(fig_ts, width='stretch')
        
        with tab3:
            st.subheader("Load Optimization Simulation")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=load_factor * 100,
                title={'text': "Load Factor (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "red"},
                                 {'range': [50, 75], 'color': "yellow"},
                                 {'range': [75, 100], 'color': "green"}]}))
            st.plotly_chart(fig_gauge, width='stretch')
            
            load_shift = st.slider("Load Shifting Potential (%)", 0, 30, 15)
            
            optimized_load = df['load'].copy()
            peak_threshold = optimized_load.quantile(0.85)
            peak_mask = optimized_load > peak_threshold
            optimized_load[peak_mask] = optimized_load[peak_mask] * (1 - load_shift/100)
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=df.index, y=df['load'], name='Original Load', line=dict(color='blue')))
            fig_opt.add_trace(go.Scatter(x=df.index, y=optimized_load, name='Optimized Load', line=dict(color='green')))
            fig_opt.update_layout(title="Load Optimization Impact Simulation")
            st.plotly_chart(fig_opt, width='stretch')
            
            savings = (df['load'].sum() - optimized_load.sum()) / 1000
            reduction_percent = ((df['load'].sum() - optimized_load.sum()) / df['load'].sum()) * 100
            st.success(f"💰 Estimated Energy Savings: {savings:.2f} MWh ({reduction_percent:.1f}% reduction)")
        
        with tab4:
            st.subheader("AI-Powered Load Optimization Insights")
            
            with st.spinner("Generating AI recommendations..."):
                llm_output = llm_analysis(df, peak_load, avg_load, load_factor, renewable_percentage, client)
                st.markdown("### 🤖 AI Analysis & Recommendations")
                st.markdown(llm_output)
        
        with tab5:
            st.subheader("Generate Report")
            
            if st.button("📊 Download CSV Report"):
                stats = {
                    "Peak Load (MW)": peak_load,
                    "Average Load (MW)": avg_load,
                    "Load Factor (%)": load_factor * 100,
                    "Renewable Share (%)": renewable_percentage,
                    "Total Records": len(df),
                    "Time Range Start": str(df.index.min()),
                    "Time Range End": str(df.index.max())
                }
                
                report_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
                csv = report_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="electricity_report.csv">\
                        <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">\
                        📥 Download CSV Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("✅ Report ready for download!")
        
        warnings.filterwarnings('default')
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

else:
    st.info("👈 Please upload your Electricity Market Dataset to begin optimization")
    
    with st.expander("📋 Dataset Information"):
        st.markdown("""
        Your Electricity Market Dataset contains columns including:
        
        **Key Columns:**
        - ✅ **Timestamp** - Date and time
        - ✅ **TotalDemand (MW)** - Electricity load
        - ✅ **WindGeneration (MW)** - Wind power
        - ✅ **HydroGeneration (MW)** - Hydro power
        - ✅ **SolarGeneration (MW)** - Solar power
        - ✅ **Temperature (°C)** - Weather data
        - ✅ **Humidity (%)** - Weather data
        - ✅ **WindSpeed (m/s)** - Weather data
        - ✅ **Price (RMB/MWh)** - Electricity price
        
        **The app provides:**
        1. Load pattern analysis with hourly/daily patterns
        2. Weather impact correlation analysis
        3. Renewable generation insights
        4. Optimization recommendations with savings estimates
        5. AI-powered policy suggestions (with Groq API)
        """)