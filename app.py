import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Latin America Historical Data Analysis - Lil Uzi Vert Edition 🌀",
    page_icon="🌀",
    layout="wide"
)

# Custom Purple Lil Uzi Vert Theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #8B008B, #9400D3, #6A0DAD, #4B0082);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #4B0082, #6A0DAD);
    }
    
    /* Headers and text */
    h1, h2, h3 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-family: 'Arial Black', Arial, sans-serif;
    }
    
    /* Main content area */
    .main .block-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 215, 0, 0.3);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FF00FF, #9400D3);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #8B008B, #FF1493);
        color: white;
        border-radius: 15px;
        font-weight: bold;
    }
    
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background-color: rgba(139, 0, 139, 0.3);
        border: 2px solid #FFD700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #4B0082, #8B008B);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FFD700;
        font-weight: bold;
    }
    
    /* Metrics and info boxes */
    .stMetric {
        background: rgba(255, 215, 0, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Success/warning messages */
    .stSuccess {
        background: linear-gradient(45deg, #32CD32, #228B22);
        color: white;
        border-radius: 10px;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #4B0082;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
COUNTRIES = {
    'Brazil': 'BRA',
    'Mexico': 'MEX', 
    'Argentina': 'ARG'
}

DATA_CATEGORIES = {
    'Population': {'indicator': 'SP.POP.TOTL', 'unit': 'people'},
    'Unemployment rate': {'indicator': 'SL.UEM.TOTL.ZS', 'unit': 'percentage'},
    'Education levels': {'indicator': 'SE.TER.ENRR', 'unit': 'tertiary enrollment rate %'},
    'Life expectancy': {'indicator': 'SP.DYN.LE00.IN', 'unit': 'years'},
    'Average income': {'indicator': 'NY.GDP.PCAP.CD', 'unit': 'USD per capita'},
    'Birth rate': {'indicator': 'SP.DYN.CBRT.IN', 'unit': 'per 1,000 people'},
    'Immigration out of country': {'indicator': 'SM.POP.NETM', 'unit': 'net migration'},
    'Murder Rate': {'indicator': 'VC.IHR.PSRC.P5', 'unit': 'per 100,000 people'}
}

# Cache data fetching
@st.cache_data(ttl=3600)
def fetch_world_bank_data(indicator, country_codes, start_year=1950, end_year=2023):
    """Fetch data from World Bank API"""
    all_data = []
    
    for country_code in country_codes:
        url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
        params = {
            'date': f"{start_year}:{end_year}",
            'format': 'json',
            'per_page': 1000
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1]:
                    for item in data[1]:
                        if item['value'] is not None:
                            all_data.append({
                                'country': item['country']['value'],
                                'year': int(item['date']),
                                'value': float(item['value'])
                            })
        except Exception as e:
            st.warning(f"Error fetching data for {country_code}: {str(e)}")
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def create_polynomial_model(x, y, degree):
    """Create polynomial regression model"""
    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(x.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_pred = model.predict(x_poly)
    r2 = r2_score(y, y_pred)
    
    # Get coefficients for equation display
    coefficients = model.coef_
    intercept = model.intercept_
    
    return model, poly_features, y_pred, r2, coefficients, intercept

def get_polynomial_equation(coefficients, intercept, degree, x_label="x"):
    """Generate polynomial equation string"""
    equation = f"y = "
    terms = []
    
    # Add coefficient terms (excluding the constant term which is at index 0)
    for i in range(degree, 0, -1):
        coef = coefficients[i]
        if abs(coef) > 1e-10:  # Only include non-zero terms
            if i == 1:
                term = f"{coef:.4f}{x_label}"
            else:
                term = f"{coef:.4f}{x_label}^{i}"
            terms.append(term)
    
    # Add intercept
    if abs(intercept) > 1e-10:
        terms.append(f"{intercept:.4f}")
    
    equation += " + ".join(terms).replace("+ -", "- ")
    return equation

def analyze_polynomial_function(coefficients, intercept, degree, x_min, x_max, country, category, unit):
    """Perform mathematical analysis of polynomial function"""
    analysis = []
    
    # Create derivative coefficients for analysis
    if degree > 1:
        derivative_coefs = []
        for i in range(1, degree + 1):
            derivative_coefs.append(coefficients[i] * i)
        
        # Find critical points by solving derivative = 0
        if degree > 2:
            # For higher degree polynomials, we'll sample points to find approximate extrema
            x_range = np.linspace(x_min, x_max, 1000)
            
            # Calculate derivative values
            derivative_values = np.zeros_like(x_range)
            for i, coef in enumerate(derivative_coefs):
                derivative_values += coef * (x_range ** i)
            
            # Find sign changes in derivative (approximate critical points)
            sign_changes = []
            for i in range(len(derivative_values) - 1):
                if derivative_values[i] * derivative_values[i + 1] < 0:
                    sign_changes.append(x_range[i])
            
            # Analyze critical points
            for critical_point in sign_changes:
                year = int(critical_point)
                
                # Calculate function value at critical point
                func_value = intercept
                for i in range(1, degree + 1):
                    func_value += coefficients[i] * (critical_point ** i)
                
                # Determine if max or min by checking second derivative sign
                second_derivative = 0
                if len(derivative_coefs) > 1:
                    for i in range(1, len(derivative_coefs)):
                        second_derivative += derivative_coefs[i] * i * (critical_point ** (i - 1))
                
                if second_derivative < 0:
                    analysis.append(f"📈 **Local Maximum**: The {category.lower()} of {country} reached a local maximum around {year}, with a value of approximately {func_value:.2f} {unit}.")
                elif second_derivative > 0:
                    analysis.append(f"📉 **Local Minimum**: The {category.lower()} of {country} reached a local minimum around {year}, with a value of approximately {func_value:.2f} {unit}.")
    
    # Find periods of fastest increase/decrease
    x_sample = np.linspace(x_min, x_max, 100)
    derivatives = []
    
    for x in x_sample:
        derivative = 0
        for i in range(1, degree + 1):
            derivative += coefficients[i] * i * (x ** (i - 1))
        derivatives.append(derivative)
    
    max_derivative_idx = np.argmax(np.abs(derivatives))
    max_derivative_year = int(x_sample[max_derivative_idx])
    max_derivative_value = derivatives[max_derivative_idx]
    
    if max_derivative_value > 0:
        analysis.append(f"🚀 **Fastest Growth**: The {category.lower()} of {country} was increasing at its fastest rate around {max_derivative_year}, at approximately {abs(max_derivative_value):.2f} {unit} per year.")
    else:
        analysis.append(f"📉 **Fastest Decline**: The {category.lower()} of {country} was decreasing at its fastest rate around {max_derivative_year}, at approximately {abs(max_derivative_value):.2f} {unit} per year.")
    
    return analysis

def get_historical_context(country, category, year, value):
    """Provide historical context for significant changes"""
    contexts = {
        'Brazil': {
            'Population': {
                range(1950, 1970): "Post-WWII population boom and urbanization",
                range(1970, 1985): "Economic miracle period driving internal migration",
                range(1985, 1995): "Democratic transition and economic instability",
                range(1995, 2010): "Economic stabilization and social programs",
                range(2010, 2023): "Economic slowdown affecting population growth"
            },
            'Life expectancy': {
                range(1950, 1970): "Public health improvements and healthcare expansion",
                range(1970, 1990): "Healthcare system development during military rule",
                range(1990, 2010): "Universal healthcare system (SUS) implementation",
                range(2010, 2023): "Continued healthcare improvements despite economic challenges"
            }
        },
        'Mexico': {
            'Population': {
                range(1950, 1970): "Post-revolution population recovery and growth",
                range(1970, 1990): "Oil boom era demographic expansion",
                range(1990, 2010): "NAFTA economic integration effects",
                range(2010, 2023): "Demographic transition and migration patterns"
            }
        },
        'Argentina': {
            'Average income': {
                range(1950, 1970): "Peronist era economic policies",
                range(1970, 1990): "Military rule and economic instability",
                range(1990, 2001): "Neoliberal reforms and convertibility plan",
                range(2001, 2015): "Post-crisis recovery and commodity boom",
                range(2015, 2023): "Economic adjustment and inflation challenges"
            }
        }
    }
    
    if country in contexts and category in contexts[country]:
        for period, context in contexts[country][category].items():
            if year in period:
                return context
    
    return "Historical data point - specific context not available"

def main():
    # Header with Lil Uzi Vert theme - using emoji instead of local image for deployment
    st.markdown("""
    <div style="text-align: center;">
        <div style="font-size: 8rem; margin: 20px 0;">
            🎤🌀💎
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #FFD700; font-size: 3rem; text-shadow: 3px 3px 6px rgba(0,0,0,0.8);">
            🌀 LIL UZI DATA VIS 🌀
        </h1>
        <h2 style="color: #FF69B4; font-size: 2rem;">
            Latin America Historical Analysis
        </h2>
        <p style="color: #FFD700; font-size: 1.3rem; font-weight: bold;">
            Created by: Charles Carter
        </p>
        <p style="color: #DDA0DD; font-size: 1.1rem; font-style: italic;">
            🎵 Money Longer - but make it DATA! Analyzing 70+ years of Latin American statistics 🎵
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("🌀 Uzi's Analysis Config 🌀")
    
    # File Downloads Section
    st.sidebar.subheader("💎 Download Files 💎")
    
    # Read app.py content for download
    with open('app.py', 'r') as f:
        app_py_content = f.read()
    
    # Read requirements content for download
    with open('requirements_for_download.txt', 'r') as f:
        requirements_content = f.read()
    
    # Download buttons
    st.sidebar.download_button(
        label="📥 Download app.py",
        data=app_py_content,
        file_name="app.py",
        mime="text/plain"
    )
    
    st.sidebar.download_button(
        label="📥 Download requirements.txt",
        data=requirements_content,
        file_name="requirements.txt",
        mime="text/plain"
    )
    
    st.sidebar.markdown("---")
    
    # Category selection
    selected_category = st.sidebar.selectbox("Select Data Category:", list(DATA_CATEGORIES.keys()))
    
    # Country selection
    country_options = ['Single Country'] + list(COUNTRIES.keys()) + ['Compare All Countries']
    selected_countries = st.sidebar.multiselect("Select Countries:", list(COUNTRIES.keys()), default=['Brazil'])
    
    # Polynomial degree
    poly_degree = st.sidebar.slider("Polynomial Degree:", 3, 6, 3)
    
    # Time increment for graph
    time_increment = st.sidebar.slider("Graph Time Increments (years):", 1, 10, 5)
    
    # Extrapolation years
    extrapolation_years = st.sidebar.slider("Extrapolation Years into Future:", 0, 50, 10)
    
    if not selected_countries:
        st.warning("Please select at least one country to analyze.")
        return
    
    # Fetch data
    with st.spinner("Fetching historical data..."):
        indicator = DATA_CATEGORIES[selected_category]['indicator']
        unit = DATA_CATEGORIES[selected_category]['unit']
        country_codes = [COUNTRIES[country] for country in selected_countries]
        
        df = fetch_world_bank_data(indicator, country_codes)
    
    if df.empty:
        st.error("No data available for the selected category and countries. Please try a different selection.")
        return
    
    # Filter to last 70 years
    current_year = datetime.now().year
    start_year = current_year - 70
    df = df[df['year'] >= start_year]
    
    # Main analysis
    st.header(f"📈 {selected_category} Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💎 Data Visualization", "🔍 Uzi's Analysis", "🔮 Future Predictions", "📋 Raw Data Table", "🖨️ Print The Money"])
    
    with tab1:
        # Create visualization
        fig = make_subplots(rows=1, cols=1)
        
        models = {}
        equations = {}
        
        for country in selected_countries:
            country_data = df[df['country'] == country].sort_values('year')
            
            if len(country_data) < 5:  # Need minimum data points
                st.warning(f"Insufficient data for {country} ({len(country_data)} points). Skipping analysis.")
                continue
            
            x = country_data['year'].values
            y = country_data['value'].values
            
            # Create polynomial model
            model, poly_features, y_pred, r2, coefficients, intercept = create_polynomial_model(x, y, poly_degree)
            models[country] = (model, poly_features, r2)
            
            # Generate equation
            equation = get_polynomial_equation(coefficients, intercept, poly_degree, "year")
            equations[country] = equation
            
            # Plot scatter points
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers',
                name=f'{country} (Data)',
                marker=dict(size=8)
            ))
            
            # Plot regression curve
            x_smooth = np.linspace(np.min(x), np.max(x), 100)
            x_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
            y_smooth = model.predict(x_smooth_poly)
            
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=f'{country} (Regression)',
                line=dict(width=3)
            ))
            
            # Add extrapolation if requested
            if extrapolation_years > 0:
                future_years = np.linspace(np.max(x), np.max(x) + extrapolation_years, 50)
                future_poly = poly_features.transform(future_years.reshape(-1, 1))
                future_pred = model.predict(future_poly)
                
                fig.add_trace(go.Scatter(
                    x=future_years,
                    y=future_pred,
                    mode='lines',
                    name=f'{country} (Projection)',
                    line=dict(width=3, dash='dash'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title=f'🌀 {selected_category} - Uzi\'s Data Analysis (Degree {poly_degree}) 🌀',
            xaxis_title='Year 📅',
            yaxis_title=f'{selected_category} ({unit}) 📊',
            height=600,
            plot_bgcolor='rgba(75, 0, 130, 0.1)',
            paper_bgcolor='rgba(139, 0, 139, 0.2)',
            font=dict(color='#FFD700'),
            title_font=dict(size=20, color='#FF69B4'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display equations
        st.subheader("🌀 Uzi's Mathematical Equations 🌀")
        for country, equation in equations.items():
            r2 = models[country][2]
            st.write(f"**{country}**: {equation}")
            st.write(f"*R² = {r2:.4f}*")
            st.write("")
    
    with tab2:
        st.subheader("💎 Uzi's Mathematical Analysis 💎")
        
        for country in selected_countries:
            if country in models:
                st.write(f"## {country}")
                
                country_data = df[df['country'] == country].sort_values('year')
                x = country_data['year'].values
                y = country_data['value'].values
                
                model, poly_features, r2 = models[country]
                coefficients = model.coef_
                intercept = model.intercept_
                
                # Perform function analysis
                analysis = analyze_polynomial_function(
                    coefficients, intercept, poly_degree, 
                    np.min(x), np.max(x), country, selected_category, unit
                )
                
                for point in analysis:
                    st.write(point)
                
                # Domain and Range
                st.write(f"📏 **Domain**: {int(np.min(x))} to {int(np.max(x))} (years in dataset)")
                st.write(f"📏 **Range**: {np.min(y):.2f} to {np.max(y):.2f} {unit}")
                
                # Historical context for significant points
                st.write("### 🏛️ Historical Context")
                mid_year = int((np.min(x) + np.max(x)) / 2)
                mid_value = float(np.mean(y))
                context = get_historical_context(country, selected_category, mid_year, mid_value)
                st.write(f"*{context}*")
                
                st.write("---")
    
    with tab3:
        st.subheader("🔮 Future Money Predictions 🔮")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Interpolation/Extrapolation")
            target_year = st.number_input("Enter year for prediction:", 
                                        min_value=1950, 
                                        max_value=2100, 
                                        value=2030)
            
            selected_country_pred = st.selectbox("Select country for prediction:", selected_countries)
            
            if st.button("Calculate Prediction"):
                if selected_country_pred in models:
                    model, poly_features, r2 = models[selected_country_pred]
                    
                    # Make prediction
                    year_poly = poly_features.transform([[target_year]])
                    prediction = model.predict(year_poly)[0]
                    
                    max_year = df[df['country'] == selected_country_pred]['year'].max()
                    prediction_type = "Interpolation" if target_year <= max_year else "Extrapolation"
                    
                    st.success(f"**{prediction_type} Result**: In {target_year}, the {selected_category.lower()} of {selected_country_pred} is predicted to be **{prediction:.2f} {unit}**")
                    
                    if prediction_type == "Extrapolation":
                        st.warning("⚠️ This is an extrapolation beyond the historical data range. Results should be interpreted with caution.")
        
        with col2:
            st.write("### Average Rate of Change")
            year1 = st.number_input("Start year:", min_value=1950, max_value=2100, value=2000)
            year2 = st.number_input("End year:", min_value=1950, max_value=2100, value=2020)
            
            selected_country_rate = st.selectbox("Select country for rate calculation:", 
                                               selected_countries, 
                                               key="rate_country")
            
            if st.button("Calculate Rate of Change"):
                if selected_country_rate in models and year2 > year1:
                    model, poly_features, r2 = models[selected_country_rate]
                    
                    # Calculate values at both years
                    year1_poly = poly_features.transform([[year1]])
                    year2_poly = poly_features.transform([[year2]])
                    
                    value1 = model.predict(year1_poly)[0]
                    value2 = model.predict(year2_poly)[0]
                    
                    rate = (value2 - value1) / (year2 - year1)
                    
                    st.success(f"**Average Rate of Change**: From {year1} to {year2}, the {selected_category.lower()} of {selected_country_rate} changed at an average rate of **{rate:.4f} {unit} per year**")
    
    with tab4:
        st.subheader("💰 The Raw Data Money 💰")
        
        # Make data editable
        edited_df = st.data_editor(
            df.pivot(index='year', columns='country', values='value'),
            width='stretch',
            num_rows="dynamic"
        )
        
        st.write(f"**Data Source**: World Bank - {selected_category}")
        st.write(f"**Unit**: {unit}")
        st.write(f"**Time Period**: Last 70 years (from {start_year})")
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data as CSV",
            data=csv,
            file_name=f"{selected_category.replace(' ', '_')}_{'-'.join(selected_countries)}_data.csv",
            mime="text/csv"
        )
    
    with tab5:
        st.subheader("🖨️ Print The Money - Summary Report 🖨️")
        
        st.markdown("---")
        st.markdown(f"# Latin America Historical Data Analysis Report")
        st.markdown(f"**Category**: {selected_category}")
        st.markdown(f"**Countries**: {', '.join(selected_countries)}")
        st.markdown(f"**Analysis Date**: {datetime.now().strftime('%B %d, %Y')}")
        st.markdown(f"**Polynomial Degree**: {poly_degree}")
        
        st.markdown("## Regression Equations")
        for country, equation in equations.items():
            if country in models:
                r2 = models[country][2]
                st.markdown(f"**{country}**: {equation} (R² = {r2:.4f})")
        
        st.markdown("## Key Findings")
        for country in selected_countries:
            if country in models:
                country_data = df[df['country'] == country].sort_values('year')
                if not country_data.empty:
                    st.markdown(f"### {country}")
                    st.markdown(f"- Data range: {int(country_data['year'].min())} - {int(country_data['year'].max())}")
                    st.markdown(f"- Minimum value: {float(country_data['value'].min()):.2f} {unit}")
                    st.markdown(f"- Maximum value: {float(country_data['value'].max()):.2f} {unit}")
                    st.markdown(f"- Model R² score: {models[country][2]:.4f}")
        
        if extrapolation_years > 0:
            st.markdown(f"## Future Projections ({extrapolation_years} years)")
            for country in selected_countries:
                if country in models:
                    model, poly_features, r2 = models[country]
                    future_year = current_year + extrapolation_years
                    year_poly = poly_features.transform([[future_year]])
                    prediction = model.predict(year_poly)[0]
                    st.markdown(f"**{country}** ({future_year}): {prediction:.2f} {unit}")
        
        st.markdown("---")
        st.markdown("*Generated by Latin America Historical Data Analysis App*")

if __name__ == "__main__":
    main()
