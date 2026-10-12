# NorthStar*

**Food guided by you.** NorthStar is an AI travel-guide chatbot that recommends real places to eat and things to do in any city, without making places up.

Built as the capstone project for Kode With Klossy's 2026 AI & Machine Learning program by Divya Bhatia, Samyutha Raja, Tobi and Vishruti.

**Try it:** [huggingface.co/spaces/kode-with-klossy/4.3-groupC4-capstone](https://huggingface.co/spaces/kode-with-klossy/4.3-groupC4-capstone)

## How it works (`capstone/`)

NorthStar uses retrieval-augmented generation (RAG):

1. **Understand the request.** A large language model (Qwen 2.5, 7B Instruct, via Hugging Face Inference) turns a casual message into structured preferences: city, price, meal, food or activities, and whether the user wants more options.
2. **Retrieve real places.** The app pulls live places from the Foursquare Places API (restaurants, cafés and bakeries, or attractions), caches results to avoid repeat calls, and falls back to a saved Yelp dataset when live data is unavailable.
3. **Answer from the data only.** Only the retrieved places are passed to the model, five at a time. The model is instructed never to invent a place, and to say so when it has no data.

The chatbot streams its replies and remembers the city and request across the conversation, so "show me more" pages through new options.

## Prototype (`prototype/`)

An earlier, filter-based interface: choose a city, meal, price, cuisine, mood and dietary needs (vegetarian, vegan, halal, gluten-free), and it returns up to ten matching restaurants from a restaurant dataset in which each place is stored as a text document. Filters that would return nothing are relaxed instead of showing an empty page.

## Running it

```bash
cd capstone
pip install -r requirements.txt
export FOURSQUARE_API_KEY=your_key   # https://foursquare.com/developers
python app.py
```

Calls to the language model need a Hugging Face account (`huggingface-cli login`).

**Data:** the Yelp data files used for the fallback and the prototype are not included in this repository. They are large, and the Yelp Open Dataset terms do not allow redistribution. Download the dataset from Yelp and place the files next to `app.py` to run those parts.

Built with Python, Gradio, Hugging Face and the Foursquare Places API.
