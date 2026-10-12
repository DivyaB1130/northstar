import gradio as gr
import pandas as pd
import re
import html
import os

# ============================================================
# LOAD NORTHSTAR DATASET
# ============================================================

DATA_FILE = "northstar_rag_dataset.parquet"

try:
    business_df = pd.read_parquet(DATA_FILE)
    print("===================================")
    print("NORTHSTAR DATASET LOADED")
    print("ROWS:", len(business_df))
    print("COLUMNS:", business_df.columns.tolist())
    print("===================================")
except Exception as e:
    business_df = pd.DataFrame()
    print("===================================")
    print("ERROR LOADING DATASET")
    print(e)
    print("===================================")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value):
    """Convert a dataset value into searchable text."""
    if pd.isna(value):
        return ""
    return str(value)


def find_column(possible_names):
    """
    Find the first matching column in the dataset.
    This lets the app work even if the dataset uses
    slightly different column names.
    """
    if business_df.empty:
        return None
    columns_lower = {str(column).lower(): column for column in business_df.columns}
    for name in possible_names:
        if name.lower() in columns_lower:
            return columns_lower[name.lower()]
    return None


# ============================================================
# IDENTIFY DATASET COLUMNS
# ============================================================

NAME_COL = find_column(["name", "business_name"])
CITY_COL = find_column(["city"])
STATE_COL = find_column(["state", "state_code"])
BUSINESS_ID_COL = find_column(["business_id", "id"])
PRICE_COL = find_column(["price_range", "price"])
CUISINE_COL = find_column(["cuisine", "cuisines"])
FOOD_TYPES_COL = find_column(["food_types", "categories", "category"])
RAG_COL = find_column(["rag_document", "text", "document"])
RATING_COL = find_column(["rating", "stars", "yelp_rating"])
ADDRESS_COL = find_column(["address", "formatted_address"])

print("===================================")
print("DETECTED COLUMNS")
print("NAME:", NAME_COL)
print("CITY:", CITY_COL)
print("STATE:", STATE_COL)
print("PRICE:", PRICE_COL)
print("CUISINE:", CUISINE_COL)
print("FOOD TYPES:", FOOD_TYPES_COL)
print("RAG:", RAG_COL)
print("RATING:", RATING_COL)
print("ADDRESS:", ADDRESS_COL)
print("===================================")


# ============================================================
# GET SEARCHABLE TEXT
# ============================================================

def searchable_text(row):
    pieces = []
    for column in [NAME_COL, CUISINE_COL, FOOD_TYPES_COL, RAG_COL]:
        if column and column in row.index:
            pieces.append(safe_text(row[column]))
    return " ".join(pieces).lower()


# ============================================================
# GET RESTAURANTS FROM DATASET
# ============================================================

def get_restaurants(location):
    if business_df.empty:
        return []
    if not location:
        return []

    df = business_df.copy()
    search_location = location.strip()

    # ========================================================
    # CITY + STATE
    # Example: Newark, New Jersey
    # ========================================================
    if "," in search_location:
        city, state = [x.strip() for x in search_location.split(",", 1)]

        # Convert full state names to abbreviations
        state_names = {
            "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
            "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
            "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY",
            "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI",
            "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
            "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
            "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
            "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
            "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
            "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY"
        }

        city_lower = city.lower().strip()
        state_upper = state.upper().strip()

        # Turn "New Jersey" into "NJ"
        state_abbreviation = state_names.get(state_upper, state_upper)

        # Clean dataset city/state columns
        city_series = df[CITY_COL].fillna("").astype(str).str.strip().str.lower()
        state_series = df[STATE_COL].fillna("").astype(str).str.strip().str.upper()

        # Match BOTH city and state
        # Match the city first
        city_mask = city_series.eq(city_lower)
        df = df[city_mask]

        # If the exact city isn't found, try a partial match
        if len(df) == 0:
            df = business_df[business_df[CITY_COL].fillna("").astype(str).str.lower().str.contains(city_lower, na=False, regex=False)]

        print("LOCATION SEARCH:", search_location)
        print("CITY:", city)
        print("STATE:", state_abbreviation)
        print("MATCHED ROWS:", len(df))

    # ========================================================
    # CITY ONLY
    # Example: Philadelphia
    # ========================================================
    else:
        city_lower = search_location.lower()
        city_series = df[CITY_COL].fillna("").astype(str).str.strip().str.lower()

        # First try exact city
        exact_matches = df[city_series.eq(city_lower)]

        if len(exact_matches) > 0:
            df = exact_matches
        else:
            # Then try partial match
            df = df[city_series.str.contains(city_lower, na=False, regex=False)]

        print("LOCATION SEARCH:", search_location)
        print("CITY:", city_lower)
        print("MATCHED ROWS:", len(df))

    # ========================================================
    # BUILD RESTAURANT OBJECTS
    # ========================================================
    restaurants = []
    for _, row in df.iterrows():
        name = safe_text(row[NAME_COL]) if NAME_COL else "Unknown Restaurant"
        address = safe_text(row[ADDRESS_COL]) if ADDRESS_COL else ""
        price = None
        if PRICE_COL:
            raw_price = row[PRICE_COL]
            try:
                if not pd.isna(raw_price):
                    price = int(float(raw_price))
            except:
                price = None
        categories = searchable_text(row)
        rating = None
        if RATING_COL:
            try:
                if not pd.isna(row[RATING_COL]):
                    rating = float(row[RATING_COL])
            except:
                rating = None
        restaurants.append({"name": name, "address": address, "categories": categories, "price": price, "rating": rating, "row": row})

    return restaurants


# ============================================================
# FILTER RESTAURANTS
# ============================================================

def get_recommendations(location, meal, price, cuisine, dietary, vibe):
    restaurants = get_restaurants(location)

    if not restaurants:
        print("NO RESTAURANTS FOUND FOR LOCATION:", location)
        return []

    results = restaurants.copy()
    print("LOCATION:", location, "| INITIAL RESULTS:", len(results))

    # ========================================================
    # PRICE
    # ========================================================
    if price:
        price_map = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
        wanted_price = price_map.get(price)
        if wanted_price:
            filtered = [r for r in results if r["price"] == wanted_price]
            # Only apply price filter if it produces results.
            if filtered:
                results = filtered

    # ========================================================
    # CUISINE
    # ========================================================
    if cuisine and cuisine != "Surprise Me":
        cuisine_keywords = {
            "Asian": ["asian", "chinese", "japanese", "thai", "indian", "korean", "vietnamese", "sushi", "ramen", "filipino", "taiwanese"],
            "Italian": ["italian", "pizza", "pasta", "italian restaurant"],
            "Mexican": ["mexican", "taco", "tex-mex", "tacos"],
            "American": ["american", "burger", "bbq", "barbecue", "steak"],
            "Healthy": ["healthy", "salad", "vegetarian", "vegan", "mediterranean"]
        }
        keywords = cuisine_keywords.get(cuisine, [cuisine.lower()])
        filtered = []
        for restaurant in results:
            text = restaurant["categories"].lower()
            if any(keyword in text for keyword in keywords):
                filtered.append(restaurant)
        # IMPORTANT:
        # Actually enforce the cuisine filter.
        results = filtered

    # ========================================================
    # MEAL
    # ========================================================
    if meal:
        meal_keywords = {
            "Breakfast": ["breakfast", "brunch", "cafe", "coffee", "bakery", "bagel", "donut"],
            "Lunch": ["lunch", "sandwich", "deli", "cafe", "restaurant", "salad", "burger"],
            "Dinner": ["restaurant", "steak", "italian", "mexican", "japanese", "bar", "dinner"],
            "Late Night": ["bar", "pub", "night", "pizza", "fast food", "24 hour"]
        }
        keywords = meal_keywords.get(meal, [])
        filtered = []
        for restaurant in results:
            text = restaurant["categories"]
            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(restaurant)
        results = filtered

    # ========================================================
    # VIBE
    # ========================================================
    if vibe:
        vibe_keywords = {
            "Quick & Casual": ["fast food", "takeout", "deli", "sandwich", "cafe", "quick"],
            "Sit-down": ["restaurant", "bistro", "steakhouse", "italian", "mexican", "dining"],
            "Special Occasion": ["steakhouse", "fine dining", "wine", "restaurant", "upscale"],
            "Café / Relaxed": ["cafe", "coffee", "bakery", "tea", "dessert"],
            "Fun & Social": ["bar", "pub", "brewery", "sports bar", "nightlife"]
        }
        keywords = vibe_keywords.get(vibe, [])
        filtered = []
        for restaurant in results:
            text = restaurant["categories"]
            if any(keyword.lower() in text for keyword in keywords):
                filtered.append(restaurant)
        if filtered:
            results = filtered

    # ========================================================
    # DIETARY
    # ========================================================
    if dietary:
        dietary_keywords = []
        for item in dietary:
            if item == "Vegetarian":
                dietary_keywords += ["vegetarian", "vegan"]
            elif item == "Vegan":
                dietary_keywords += ["vegan"]
            elif item == "Halal":
                dietary_keywords += ["halal"]
            elif item == "Gluten-Free":
                dietary_keywords += ["gluten free", "gluten-free"]

        if dietary_keywords:
            filtered = []
            for restaurant in results:
                text = restaurant["categories"]
                if any(keyword in text for keyword in dietary_keywords):
                    filtered.append(restaurant)
            if filtered:
                results = filtered

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================
    seen = set()
    unique_results = []
    for restaurant in results:
        key = restaurant["name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_results.append(restaurant)
    results = unique_results

    # ========================================================
    # RANDOMIZE
    # ========================================================
    if len(results) > 10:
        results = pd.DataFrame(results).sample(n=min(20, len(results)), random_state=None).to_dict("records")

    print("FINAL RESULTS:", len(results))
    return results[:10]


# ============================================================
# FILTER SUMMARY
# ============================================================

def build_filter_summary_html(location, meal, price, cuisine, dietary, vibe):
    dietary_text = ", ".join(dietary) if dietary else "No dietary restrictions"

    preferences = []
    if meal:
        preferences.append(meal)
    if price:
        preferences.append(price)
    if cuisine:
        preferences.append(cuisine)
    if vibe:
        preferences.append(vibe)

    preference_text = " — ".join(preferences) if preferences else "No preferences set"

    return f"""
    <div class="filter-summary">
        <p>📍 {html.escape(location or "Anywhere")}</p>
        <p>{html.escape(preference_text)}</p>
        <p>{html.escape(dietary_text)}</p>
    </div>
    """


# ============================================================
# BUILD RESULTS HTML
# ============================================================

def build_results_html(location, meal, price, cuisine, dietary, vibe):
    restaurants = get_recommendations(location, meal, price, cuisine, dietary, vibe)

    if not restaurants:
        return """
        <p class="no-results">
            No matches found — try changing your filters.
        </p>
        """

    cards = ""
    for restaurant in restaurants:
        name = restaurant.get("name", "Unknown")
        rag = str(restaurant.get("rag_document", ""))

        # -----------------------------------------
        # EXTRACT RATING
        # -----------------------------------------
        rating_match = re.search(r"yelp rating:\s*([\d.]+)", rag, re.IGNORECASE)
        rating = rating_match.group(1) if rating_match else "N/A"

        # -----------------------------------------
        # EXTRACT PRICE
        # -----------------------------------------
        price_match = re.search(r"price tier:\s*(\d)", rag, re.IGNORECASE)
        if price_match:
            price_number = int(price_match.group(1))
            price_display = "$" * price_number
        else:
            price_display = "$$"

        # -----------------------------------------
        # EXTRACT FOOD TYPES
        # -----------------------------------------
        food_match = re.search(r"food types:\s*(.*?)(?:\s+yelp rating:|\s+price tier:|\s+customer insights:|$)", rag, re.IGNORECASE)
        food_types = food_match.group(1).strip() if food_match else "Restaurant"

        # -----------------------------------------
        # EXTRACT LOCATION
        # -----------------------------------------
        location_match = re.search(r"location:\s*(.*?)(?:\s+food types:|\s+yelp rating:|\s+price tier:|$)", rag, re.IGNORECASE)
        restaurant_location = location_match.group(1).strip() if location_match else (location or "Location not listed")

        # -----------------------------------------
        # CLEAN UP FOOD TYPES
        # -----------------------------------------
        food_types = food_types.replace("_", " ")

        # Capitalize each category
        food_types_display = ", ".join(word.strip().title() for word in food_types.split(",") if word.strip())

        # -----------------------------------------
        # BUILD CARD
        # -----------------------------------------
        cards += f"""
        <div class="result-card">

            <h3>{name}</h3>

            <p class="result-meta">
                ⭐ {rating} &nbsp;—&nbsp; {price_display}
            </p>

            <p class="result-address">
                📍 {restaurant_location}
            </p>

            <p class="result-tagline">
                {food_types_display}
            </p>

        </div>
        """

    return cards

# ============================================================
# BUTTON FUNCTIONS
# ============================================================

def handle_find_food(location, meal, price, cuisine, dietary, vibe):
    summary = build_filter_summary_html(location, meal, price, cuisine, dietary, vibe)
    results = build_results_html(location, meal, price, cuisine, dietary, vibe)
    return (gr.update(visible=False), gr.update(visible=True), summary, results)


def handle_edit_filters():
    return (gr.update(visible=True), gr.update(visible=False))


# ============================================================
# CSS
# ============================================================

custom_css = """

body,
.gradio-container {
    background-color: #C9A9A6 !important;
}

#header {
    text-align: center;
    background-color: #183B35 !important;
    color: #FAF1E6 !important;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 25px;
}

#header h1,
#header p {
    color: #FAF1E6 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #183B35 !important;
}

#location-box {
    background-color: #183B35 !important;
    border-radius: 20px !important;
    padding: 20px !important;
}

#location-box input {
    background-color: #041D47 !important;
    border: 2px solid #FAF1E6 !important;
    color: #FAF1E6 !important;
}

#location-box input::placeholder {
    color: #C9A9A6 !important;
}

#preferences-group {
    background-color: #183B35 !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

#preferences-group .form,
#preferences-group .block {
    background-color: #183B35 !important;
    border: none !important;
}

#meal-radio,
#price-radio,
#cuisine-radio,
#dietary-checkbox,
#vibe-radio {
    background-color: transparent !important;
    padding: 10px 0 !important;
}

#meal-radio label,
#price-radio label,
#cuisine-radio label,
#vibe-radio label,
#dietary-checkbox label {
    background-color: #5B2333 !important;
    color: #FAF1E6 !important;
    border-radius: 10px !important;
    border: 2px solid #5B2333 !important;
    padding: 10px 15px !important;
}

#meal-radio label span,
#price-radio label span,
#cuisine-radio label span,
#vibe-radio label span,
#dietary-checkbox label span {
    color: #FAF1E6 !important;
}

#meal-radio label:has(input:checked),
#price-radio label:has(input:checked),
#cuisine-radio label:has(input:checked),
#vibe-radio label:has(input:checked),
#dietary-checkbox label:has(input:checked) {
    background-color: #4E7360 !important;
    border-color: #4E7360 !important;
}

label {
    color: #183B35 !important;
}

#find-button {
    background-color: #5B2333 !important;
    color: #FAF1E6 !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 15px !important;
}

#find-button:hover {
    background-color: #183B35 !important;
}

#results-header-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    margin-bottom: 20px !important;
}

#results-header-row h2 {
    margin: 0 !important;
}

#edit-filters-button {
    background-color: #5B2333 !important;
    color: #FAF1E6 !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 18px !important;
}

#edit-filters-button:hover {
    background-color: #183B35 !important;
}

.filter-summary {
    background-color: #183B35 !important;
    color: #FAF1E6 !important;
    border-radius: 12px !important;
    padding: 15px 20px !important;
    margin-bottom: 20px !important;
}

.filter-summary p {
    margin: 4px 0 !important;
    color: #FAF1E6 !important;
}

.result-card {
    background-color: #F3E7DC !important;
    color: #183B35 !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin-bottom: 18px !important;
    border: 2px solid #4E7360 !important;
}

.result-card h3 {
    margin-top: 0 !important;
    color: #183B35 !important;
}

.result-meta {
    color: #5B2333 !important;
    font-weight: bold !important;
}

.result-address {
    color: #183B35 !important;
}

.result-tagline {
    font-style: italic !important;
    color: #183B35 !important;
}

.no-results {
    color: #183B35 !important;
    text-align: center !important;
    padding: 20px !important;
}

"""


# ============================================================
# GRADIO APP
# ============================================================

with gr.Blocks(title="NorthStar*") as demo:

    # ========================================================
    # HEADER
    # ========================================================
    gr.Markdown(
        """
        <div id="header">
            <h1>★ NORTHSTAR*</h1>
            <p>FOOD GUIDED BY YOU</p>
            <p>Find your next favorite bite.</p>
        </div>
        """,
        elem_id="header"
    )

    # ========================================================
    # SCREEN 1
    # ========================================================
    with gr.Column(visible=True, elem_id="screen1") as screen1:

        gr.Markdown("## Where are you eating?")

        location = gr.Textbox(label="📍 Location", placeholder="Search a city, neighborhood, or destination...", elem_id="location-box")

        gr.Markdown("## What are you looking for?")

        with gr.Group(elem_id="preferences-group"):
            meal = gr.Radio(["Breakfast", "Lunch", "Dinner", "Late Night"], label="🍽️ Meal", elem_id="meal-radio")
            price = gr.Radio(["$", "$$", "$$$", "$$$$"], label="💰 Price Range", elem_id="price-radio")
            cuisine = gr.Radio(["Asian", "Italian", "Mexican", "Healthy", "American", "Surprise Me"], label="🥗 Cuisine", elem_id="cuisine-radio")
            dietary = gr.CheckboxGroup(["Vegetarian", "Vegan", "Halal", "Gluten-Free"], label="🥒 Dietary Preferences (Optional)", elem_id="dietary-checkbox")
            vibe = gr.Radio(["Quick & Casual", "Sit-down", "Special Occasion", "Café / Relaxed", "Fun & Social"], label="✨ Dining Vibe", elem_id="vibe-radio")

        find_button = gr.Button("FIND MY FOOD →", elem_id="find-button")

    # ========================================================
    # SCREEN 2
    # ========================================================
    with gr.Column(visible=False, elem_id="screen2") as screen2:

        with gr.Row(elem_id="results-header-row"):
            gr.Markdown("## ← NorthStar*")
            edit_filters_button = gr.Button("★ Edit Filters", elem_id="edit-filters-button")

        filter_summary_display = gr.HTML()

        gr.Markdown("## Here's what I found for you!!")

        results_display = gr.HTML()

    # ========================================================
    # WIRING
    # ========================================================
    find_button.click(
        fn=handle_find_food,
        inputs=[location, meal, price, cuisine, dietary, vibe],
        outputs=[screen1, screen2, filter_summary_display, results_display]
    )

    edit_filters_button.click(fn=handle_edit_filters, inputs=[], outputs=[screen1, screen2])

print("\n===== LOCATION TEST =====")
for city in ["Philadelphia", "Newark", "Las Vegas", "Chicago"]:
    matches = business_df["city"].fillna("").astype(str).str.strip().str.lower().eq(city.lower())
    matches = business_df[matches]
    print(city, "→", len(matches), "businesses")
    if len(matches) > 0:
        print(matches["state"].value_counts().to_dict())
    print(business_df["city"].value_counts().head(25))

# ============================================================
# LAUNCH
# ============================================================

demo.launch(css=custom_css)