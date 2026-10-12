import gradio as gr
import pandas as pd
import os
import json
import requests
from huggingface_hub import InferenceClient
FOURSQUARE_API_KEY = os.environ.get("FOURSQUARE_API_KEY")

def extract_preferences(message):
    extraction_prompt = (
        "Extract the following from the user's message and return ONLY valid JSON.\n"
        "Fields: destination_city (string or null), current_location (string or null), price (integer 1-4 or null), "
        "meal (string or null), intent (one of: \"food\", \"activities\", or null if unclear), "
        "wants_more (true if the user is asking for different/additional/other recommendations, otherwise false).\n\n"
        f"User message: \"{message}\"\n\nJSON:"
    )
    response = client.chat_completion([{"role": "user", "content": extraction_prompt}])
    raw = response.choices[0].message.content.strip()
    try: #trying to covert the reply into a dictionary  
        preferences = json.loads(raw)
  #if for any reason it doesn't covert, fall back to default(all fields empty) so the whole code doesn't stop
    except json.JSONDecodeError:
        preferences = {"destination_city": None, "price": None, "meal": None, "current_location": None, "intent": None, "wants_more": False}
    return preferences
    
recommendation_cache = {}  #storage
def get_food_recommendations(city, price=None, limit=50):
    cache_key = f"{city.strip().lower()}_{price}_{limit}"
    if cache_key in recommendation_cache:
        print(f"Using cached results for '{city}' — no API call made.")
        return recommendation_cache[cache_key]
        
    url = "https://places-api.foursquare.com/places/search"
    headers = {#identifies us when we go to the url
        "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
        "Accept": "application/json",
        "X-Places-Api-Version": "2025-06-17"
    }
    '''
13065 — Restaurant (covers every restaurant sub-type), 13032 — Café / Coffee House, 13002 — Bakery, 13054 — Food Truck, 13040 — Dessert Shop
13052 — Food Court, 13053 — Food Stand)
    '''
    params = {#paramenters for the search
        "near": city,
        "fsq_category_ids": "13065,13032,13002,13145,13054,13040,13052,13053",
        "fields": "name,location,categories,price",
        "limit": limit,
    }
    response = requests.get(url, headers=headers, params=params, timeout=10)
    
    if response.status_code == 429:#lets us know if there have beeen too many pull requests instead of just crashing
        retry_after = response.headers.get("Retry-After", "unknown")
        print(f"Rate limited — retry after: {retry_after} seconds. Response text: {response.text}")
        return []        

    response.raise_for_status()
    results = response.json().get("results", [])

    chunks = []
    for place in results: #organising all the info we got
        name = place.get("name", "Unknown")
        address = place.get("location", {}).get("formatted_address", "Address not listed")
        category = ", ".join(c["name"] for c in place.get("categories", []))
        place_price = place.get("price", "Price not listed")
        chunks.append(f"Restaurant: {name} | Address: {address} | Categories: {category} | Price: {place_price}")
    recommendation_cache[cache_key] = chunks # storing here
    return chunks
    
attraction_cache = {} #storage
def get_activity_recommendations(city, limit=50):
    cache_key = f"{city.strip().lower()}_{limit}"
    if cache_key in attraction_cache:
        print(f"Using cached results for '{city}' — no API call made.")
        return attraction_cache[cache_key]

    url = "https://places-api.foursquare.com/places/search"
    headers = { #identifies us when we vist the url to pull info
        "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
        "Accept": "application/json",
        "X-Places-Api-Version": "2025-06-17"
    }
    '''
foursquare category ids: 10000 -	Arts and Entertainment, 16000 - Landmarks and Outdoors, 18000- Sports and Recreation, 
19000 - Travel and Transportation, 17116 - Souvenir Store, 17144 - Market, 17115 - Shopping Plaza, 17089 - Gift Store.
    '''
    params = {#the parameters for the search we want to do
        "near": city,
        "fsq_category_ids": "10000,16000,18000,19000,17116,17144,17115,17089,17002,17138,17018",
        "fields": "name,location,categories,price",
        "limit": limit,
    }
    
    response = requests.get(url, headers=headers, params=params, timeout=10)
    if response.status_code == 429:#lets us know if there have beeen too many pull requests instead of just crashing
        retry_after = response.headers.get("Retry-After", "unknown")
        print(f"Rate limited — retry after: {retry_after} seconds. Response text: {response.text}")
        return []     

    response.raise_for_status()
    results = response.json().get("results", [])

    chunks = []
    for place in results: #organising all the info we got
        name = place.get("name", "Unknown")
        address = place.get("location", {}).get("formatted_address", "Address not listed")
        category = ", ".join(c["name"] for c in place.get("categories", []))
        place_price = place.get("price", "Price not listed")
        chunks.append(f"Attraction: {name} | Address: {address} | Categories: {category} | Price: {place_price} ")
    attraction_cache[cache_key] = chunks #storing info here
    return chunks

business_df = pd.read_json("yelp_academic_dataset_business.json", lines=True)
for col in ["name", "city", "categories"]:
    business_df[col] = business_df[col].fillna("")
business_df["stars"] = business_df["stars"].fillna("No rating available")
business_df = business_df[business_df["categories"].str.contains("Restaurant|Food", case=False, na=False)].copy()

def get_static_recommendations(city, limit=50):
    matches = business_df[business_df["city"].str.lower() == city.strip().lower()]
    chunks = []
    for _, row in matches.head(limit).iterrows():
        chunks.append(f"Restaurant: {row['name']} | City: {row['city']} | Categories: {row['categories']} | Rating: {row['stars']}")
    return chunks
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")

def respond(message, history_for_llm, name, location, meals, drinks, conversation_state):
    if conversation_state is None:
        conversation_state = {"offset": 0, "city": None, "intent": None}
    yield "Thinking...", conversation_state # shows immediately while the slower steps run
    preferences = extract_preferences(message)   
    # Fall back to the remembered city/intent from state if this message didn't mention one
    city = preferences.get("destination_city") or preferences.get("current_location") or location or conversation_state.get("city")
    print(f"DEBUG — location box value: '{location}' | resolved city: '{city}'") #trying to check something
    intent = preferences.get("intent") or conversation_state.get("intent") or "food"
    meal = (", ".join(meals) if meals else None) or preferences.get("meal") or "a meal"
    drink_note = f" They'd also like a {', '.join(drinks)}." if drinks else ""
    greeting_name = name.strip() if name else "there"

    wants_more = preferences.get("wants_more", False)

    # Only advance the offset if it's a "more" request for the SAME city — a new city always resets to page 1
    if wants_more and city == conversation_state.get("city"):
        offset = conversation_state.get("offset", 0) + 5
    else:
        offset = 0

    if not city:
        northstar_chunks = []
        data_status = "unavailable"
    elif intent == "activities":
        northstar_chunks = get_activity_recommendations(city=city)
        data_status = "real" if northstar_chunks else "unavailable"
    else:
        # default to food if intent is food or unclear
        northstar_chunks = get_food_recommendations(city=city,price=preferences.get("price"))
        if northstar_chunks:
            data_status = "real"
        else:
            northstar_chunks = get_static_recommendations(city=city)
            data_status = "static" if northstar_chunks else "unavailable"

    page = northstar_chunks[offset : offset + 5] # Save everything back into state for next turn
    conversation_state = {"offset": offset, "city": city, "intent": intent}
    if not page:
        results_text = "No results available. Do not invent any places — tell the user you don't have data for this yet."
        data_status = "unavailable"
    else:
        results_text = str(page)
 
    system_instruction = (
    "You are NorthStar*, a friendly AI travel guide for food and activities. "
    f"The user's name is {greeting_name}.{drink_note} "
    f"The user is looking for: {meal}. "
    f"Data availability status: {data_status}. "
    "If the data status is 'unavailable', clearly tell the user you're temporarily unable to pull live results "
    "and ask them to try again shortly — do NOT invent, brainstorm, or suggest any specific place names, addresses, or businesses. "
    "If the data status is 'real', recommend up to five places that genuinely fit the meal or activity type, using only what's listed below - "
    "if fewer than five relevant places are available in the data, only list the ones that are actually there. "
    "If the data status is 'static', tell the user clearly that live results aren't available right now,"
    "Ask them if they want to see saved reults. If they agree, make sure to tell them you're showing results from a saved dataset instead — mention it may not reflect current hours, prices, "
    "or whether the business is still open, and suggest they check the restaurant's own website or search engine for up-to-date details."
    "Never invent additional places, branches, or variations. "
    f"Here is the live data (if any): {results_text}"
)
           
    #managing the conversation memory
    messages = [{"role":"system","content": system_instruction}]

    #include past conversation logs
    if history_for_llm:
        messages.extend(history_for_llm)

    #include the current user message
    messages.append({"role":"user","content":message})
    
    #get the answer and launching the app
    stream = client.chat_completion(messages, max_tokens=600, temperature=0.2, stream=True)
    #gives the chatbot the appearance of typing
    partial_response = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            partial_response += token
            yield partial_response, conversation_state

custom_css = """
.gradio-container {
    background-color: #FAF7F2;
}
label {
    font-weight: 600;
    color: #3A3A3A;
}
"""
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue")) as chatbot:
    
    gr.Image("Northstar_logo.png", show_label=False, container=False, height=250)
    
    gr.Markdown("# NorthStar")
    gr.Markdown("Hi! I'm NorthStar — your guide to finding great food and things to do wherever you're headed.")
    
    with gr.Row():
        name_input = gr.Textbox(label="Your name", placeholder="e.g. Alex")
        location_input = gr.Textbox(label="Your current location", placeholder="e.g. Austin, TX")
    
    with gr.Row():
        meal_input = gr.CheckboxGroup(choices=["Breakfast", "Brunch", "Lunch", "Dinner", "Snack"], label="Meal of the day")
        drink_input = gr.CheckboxGroup(choices=["Coffee", "Soda", "Smoothie", "Tea"], label="Drink (optional)")

    send_button = gr.Button("Send")
    
    conversation_state = gr.State(None)
    chat_display = gr.Chatbot(label="NorthStar Chat")
    msg_input = gr.Textbox(label="Your message", placeholder="Tell me want you want then press enter")
    
    def user_submit(message, chat_history):
        chat_history = chat_history or []
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": ""})  # placeholder, filled in as the bot streams
        return "", chat_history

    def bot_reply(chat_history, name, location, meals, drinks, conversation_state):
        message = chat_history[-2]["content"]     # the user's message (second-to-last entry)
        history_for_llm = chat_history[:-2]        # everything before this exchange — already role/content shaped
    
        for partial, new_state in respond(message, history_for_llm, name, location, meals, drinks, conversation_state):
            chat_history[-1]["content"] = partial   # update the assistant placeholder's content
            yield chat_history, new_state

    msg_input.submit(user_submit, [msg_input, chat_display], [msg_input, chat_display]).then(
        bot_reply, [chat_display, name_input, location_input, meal_input, drink_input, conversation_state],
        [chat_display, conversation_state]
    )
    send_button.click(user_submit, [msg_input, chat_display], [msg_input, chat_display]).then(
        bot_reply, [chat_display, name_input, location_input, meal_input, drink_input, conversation_state],
        [chat_display, conversation_state]
    )

'''
conversation_state = gr.State(None)  # renamed from results_offset for clarity, since it's holding more than just an offset now
chatbot = gr.ChatInterface(
    respond,
    additional_inputs=[name_input, location_input, meal_input, drink_input, conversation_state],
    additional_outputs=[conversation_state],
    title="🧭 NorthStar",
    description="Hi! I'm NorthStar — your guide to finding great food and things to do wherever you're headed. What are you in the mood for?"
) #commenting all this out while i try the style changes
''' 
chatbot.launch(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css)

