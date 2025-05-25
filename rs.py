import streamlit as st
import ee
import geemap.foliumap as geemap
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Earth Engine (GEE)
def initialize_ee():
    try:
        # Try to authenticate and initialize Earth Engine
        ee.Initialize(project='....')
    except Exception:
        # If authentication fails, prompt for authentication
        ee.Authenticate()
        ee.Initialize(project='.....')

# Initialize Earth Engine on app start
initialize_ee()

# Streamlit Page Setup (sets up the layout and title for the app)
st.set_page_config(layout="wide")
st.title("🌍 GES-Coastal Monitor – Mediterranean Environmental Status")

# Sidebar Parameters (interactive controls for the user to choose)
with st.sidebar:
    st.header("⚙️ Analysis Parameters")
    
    # Lists of African and Levant countries for easy selection
    AFRICAN_COUNTRIES = ["Algeria", "Morocco", "Tunisia", "Libya", "Egypt"]
    LEVANT_COUNTRIES = ["Lebanon", "Syria"]
    ALL_COUNTRIES = AFRICAN_COUNTRIES + LEVANT_COUNTRIES
    
    # Dropdown for selecting a country
    country_name = st.selectbox("Select Country", ALL_COUNTRIES)

    # Default buffer distance for coastal analysis, smaller for larger countries
    default_buffer_km = 5 if country_name not in ["Egypt", "Algeria"] else 3
    buffer_km = st.slider("Coastal Buffer (km)", 1, 50, default_buffer_km)

    # Define the range of years for analysis (from 2000 to current year)
    MIN_YEAR = 2000
    MAX_YEAR = 2023

    # User inputs for start and end years for the analysis period
    start_year = st.number_input("Start Year", MIN_YEAR, MAX_YEAR, 2010)
    end_year = st.number_input("End Year", MIN_YEAR, MAX_YEAR, 2015)

    # Check for invalid year range (start year must not be greater than end year)
    if start_year > end_year:
        st.error("Start year must be less than or equal to end year.")

    # Options to choose NDVI and LST data products
    ndvi_product = st.selectbox("NDVI Product", ["MOD13A1", "MOD13Q1"])
    lst_product = st.selectbox("LST Product", ["MOD11A1", "MOD21A2"])
    
    # Quick mode for faster processing (good for large countries)
    quick_mode = st.checkbox("⚡ Quick Mode (faster for big countries)", value=False)
    
    # Button to trigger export to Google Drive
    export_requested = st.button("📤 Export to Google Drive")

# Load base datasets (geospatial data needed for analysis)
countries = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
water_mask = ee.Image("JRC/GSW1_3/GlobalSurfaceWater").select('occurrence')
save_med_coastline = ee.FeatureCollection("projects/ee-mariahussien1620/assets/med_coastlinecrop_latlong")

# Mediterranean bounding box (geographical limits of the Mediterranean)
mediterranean = ee.Geometry.Polygon([[-5.5, 30.0], [36.0, 30.0], 
                                     [36.0, 41.0], [28.0, 42.0], 
                                     [10.0, 45.0], [-5.5, 38.0], 
                                     [-5.5, 30.0]])

# Functions for extracting and processing data

# Function to extract African coastline based on selected country and buffer
def get_african_coastline(country_name, buffer_m):
    country = countries.filter(ee.Filter.eq('ADM0_NAME', country_name)).geometry()
    med_water = water_mask.gt(90).selfMask().clip(mediterranean)
    inland = country.buffer(-500)  # Buffer inland for extraction
    coastal_strip = country.difference(inland)  # Extract coastal strip
    coastline = med_water.clip(coastal_strip)
    coastline_vectors = coastline.reduceToVectors(
        geometry=coastal_strip,
        scale=30,
        geometryType='polygon',
        eightConnected=False,
        labelProperty='coastline',
        maxPixels=1e13
    )
    coastal_geom = coastline_vectors.geometry().buffer(buffer_m).intersection(country, ee.ErrorMargin(100))
    return coastal_geom

# Function to extract Levant coastline from SaveMedCoasts dataset
def get_levant_coastline(country_name, buffer_m, quick=False):
    simplify_tol = 1000 if quick else 500  # Simplify geometry for faster processing in quick mode
    country_geom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
                     .filter(ee.Filter.eq('country_na', country_name)) \
                     .geometry().simplify(1000)  # Simplify geometry of country for boundary extraction
    coastline_geom = save_med_coastline.filterBounds(country_geom) \
                                       .geometry().intersection(country_geom, ee.ErrorMargin(100)) \
                                       .simplify(simplify_tol)
    buffer_geom = coastline_geom.buffer(buffer_m).intersection(country_geom, ee.ErrorMargin(100))
    return buffer_geom

# Function to retrieve NDVI (Normalized Difference Vegetation Index) data for a given period
def get_ndvi(start, end, product):
    collection = ee.ImageCollection(f"MODIS/006/{product}") \
        .filterDate(f"{start}-01-01", f"{end}-12-31")

    def mask_modis(image):
        qa = image.select('SummaryQA')  # Quality assurance band
        mask = qa.eq(0)  # 0 means good quality pixel (no clouds, no contamination)
        return image.updateMask(mask)  # Apply the mask

    masked_collection = collection.map(mask_modis)
    return masked_collection.select('NDVI').mean().multiply(0.0001)

# Function to retrieve LST (Land Surface Temperature) data for a given period
def get_lst(start, end, product):
    band = "LST_Day_1km"
    collection = ee.ImageCollection(f"MODIS/006/{product}") \
        .filterDate(f"{start}-01-01", f"{end}-12-31")
    
    def mask_lst(image):
        qa = image.select('QC_Day')
        mask = qa.bitwiseAnd(3).eq(0)  # Bits 0-1: 00 = good data
        return image.updateMask(mask)

    masked_collection = collection.map(mask_lst)
    return masked_collection.select(band).mean().multiply(0.02).subtract(273.15)

# Main logic for extraction, processing, and visualization
buffer_m = buffer_km * 1000  # Convert km to meters

# Depending on the country selected, use different coastline extraction methods
if country_name in AFRICAN_COUNTRIES:
    st.info(f"🌍 Extracting strict coastline for {country_name}...")
    coastal_zone = get_african_coastline(country_name, buffer_m)
else:
    st.info(f"🌍 Using SaveMedCoasts coastline for {country_name}...")
    coastal_zone = get_levant_coastline(country_name, buffer_m, quick=quick_mode)

# Retrieve NDVI and LST images based on user inputs
ndvi_img = get_ndvi(start_year, end_year, ndvi_product)
lst_img = get_lst(start_year, end_year, lst_product)

# Normalize NDVI and LST values for GES (Good Environmental Status)
ndvi_min = 0.1
ndvi_max = 0.8
ndvi_norm = ndvi_img.subtract(ndvi_min).divide(ndvi_max - ndvi_min).multiply(100)

lst_min = 15
lst_max = 45
lst_norm = ee.Image(lst_max).subtract(lst_img)  # Correct Earth Engine subtraction for LST
lst_norm = lst_norm.divide(lst_max - lst_min).multiply(100)

# Calculate GES as an average of normalized NDVI and LST
ges = ndvi_norm.multiply(0.5).add(lst_norm.multiply(0.5)).rename('GES')

# Classify GES based on thresholds (80-60-40-20)
classified = ges.expression(
    "(b('GES') >= 80) ? 4 : "
    "(b('GES') >= 60) ? 3 : "
    "(b('GES') >= 40) ? 2 : "
    "(b('GES') >= 20) ? 1 : 0"
).rename("class").clip(coastal_zone).toByte()

# Visualization settings for the GES classification map
palette = ["#e61919", "#f4d35e", "#7a9e3a", "#8db14f", "#2b9348"]
vis_params = {'min': 0, 'max': 4, 'palette': palette}
map_scale = 1000 if quick_mode else 500

# Prepare the classified GES map for display
classified_proj = classified.setDefaultProjection(crs='EPSG:4326', scale=map_scale)
classified_vis = classified_proj.reduceResolution(reducer=ee.Reducer.mode(), bestEffort=True) \
                                .reproject(crs='EPSG:4326', scale=map_scale)

# Create a map to display the results
m = geemap.Map(center=[34.5, 18], zoom=5)
m.add_basemap("SATELLITE")
m.addLayer(classified_vis, vis_params, "GES Classification")
m.addLayer(coastal_zone, {"color": "cyan"}, "Coastal Buffer")

# Add a legend for the GES classification
legend_dict = {
    "Very Severe Degradation": "#e61919",
    "Moderate Degradation": "#f4d35e",
    "Stable": "#7a9e3a",
    "Good Improvement": "#8db14f",
    "Excellent Improvement": "#2b9348"
}

m.add_legend(
    title="GES Classification",
    legend_dict=legend_dict,
    position="bottomright"
)

# Display the map in the Streamlit app
m.to_streamlit(height=650)

try:
    hist_dict = classified.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=coastal_zone,
        scale=map_scale,
        maxPixels=1e7
    ).get('class')

    if hist_dict:
        hist = ee.Dictionary(hist_dict).getInfo()

        labels = ["Very Severe Degradation", "Moderate Degradation", "Stable", "Good Improvement", "Excellent Improvement"]
        colors = ["#e61919", "#f4d35e", "#7a9e3a", "#8db14f", "#2b9348"]

        values = [hist.get(str(i), 0) for i in range(5)]
        total = sum(values)
        percentages = [(v / total) * 100 if total > 0 else 0 for v in values]

        # Plot a bar chart with percentages
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, percentages, color=colors, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('Pixel Percentage (%)')
        ax.set_title('GES Class Distribution (%)', fontsize=10)
        ax.set_xticklabels(labels, rotation=10, ha='right')

        st.pyplot(fig)

        # Add GES classification table with percentages
        ges_table = pd.DataFrame({
            "GES Class": labels,
            "Pixel Percentage (%)": [f"{p:.2f}%" for p in percentages]
        })

        st.subheader("📋 GES Classification Table")
        st.dataframe(ges_table, use_container_width=True)

    else:
        st.warning("⚠️ No valid GES data for selected region.")
except Exception as e:
    st.warning(f"⚠️ Could not generate histogram: {e}")


# Time series analysis for NDVI and LST

years = list(range(start_year, end_year + 1))

ndvi_list = []
lst_list = []

# Fetch NDVI and LST data for each year in the range
for year in years:
    ndvi_year = get_ndvi(year, year, ndvi_product)
    lst_year = get_lst(year, year, lst_product)

    # Calculate the mean NDVI and LST for the coastal zone
    ndvi_mean = ndvi_year.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=coastal_zone,
        scale=map_scale,
        maxPixels=1e7
    ).get('NDVI')

    lst_mean = lst_year.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=coastal_zone,
        scale=map_scale,
        maxPixels=1e7
    ).get('LST_Day_1km')

    ndvi_list.append(ndvi_mean)
    lst_list.append(lst_mean)

# Convert the values to a list (fetched from Earth Engine)
ndvi_values = ee.List(ndvi_list).getInfo()
lst_values = ee.List(lst_list).getInfo()

# Create a DataFrame for the time series
df_timeseries = pd.DataFrame({
    "Year": years,
    "NDVI": ndvi_values,
    "LST (°C)": lst_values
})

# Plot the combined time series for NDVI and LST
st.subheader("📈 NDVI and LST Time Series (Combined)")

fig, ax1 = plt.subplots(figsize=(10, 6))

color_ndvi = 'tab:green'
color_lst = 'tab:red'

# Plot NDVI on the left axis
ax1.set_xlabel('Year')
ax1.set_ylabel('NDVI', color=color_ndvi)
ax1.plot(df_timeseries["Year"], df_timeseries["NDVI"], color=color_ndvi, marker='o', label='NDVI')
ax1.tick_params(axis='y', labelcolor=color_ndvi)
ax1.set_ylim(0, 1)

# Create a second Y-axis for LST
ax2 = ax1.twinx() 
ax2.set_ylabel('LST (°C)', color=color_lst)
ax2.plot(df_timeseries["Year"], df_timeseries["LST (°C)"], color=color_lst, marker='o', label='LST (°C)')
ax2.tick_params(axis='y', labelcolor=color_lst)

fig.tight_layout()
st.pyplot(fig)

# Display the NDVI and LST time series table
st.subheader("📋 NDVI and LST Table")
st.dataframe(df_timeseries, use_container_width=True)

# Option to download the CSV file of the time series
csv = df_timeseries.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download NDVI-LST Table as CSV",
    data=csv,
    file_name=f"{country_name}_NDVI_LST_Timeseries.csv",
    mime='text/csv'
)

# Export the GES classification image to Google Drive if the user requested
if export_requested:
    region = coastal_zone.bounds().getInfo()['coordinates']
    export_task = ee.batch.Export.image.toDrive(
        image=classified,
        description=f"GES_{country_name}",
        folder="GEE_Exports",
        scale=map_scale,
        region=region,
        fileFormat='GeoTIFF'
    )
    export_task.start()
    st.success("✅ Export started to Google Drive.")
