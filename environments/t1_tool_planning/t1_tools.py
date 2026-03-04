"""
OpenAI function definitions for T1's 14 travel planning tools.

These are passed to managed_server.chat_completion(tools=T1_TOOLS)
so the model uses proper tool calling instead of raw code generation.
"""

T1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels in a city. Requires city, checkin_date, checkout_date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "checkin_date": {"type": "array", "items": {"type": "string"}},
                    "checkout_date": {"type": "array", "items": {"type": "string"}},
                    "num_rooms": {"type": "integer"},
                    "num_people": {"type": "integer"},
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                    "hotel_name": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "integer"},
                    "rating": {"type": "array", "items": {"type": "number"}},
                    "stars": {"type": "array", "items": {"type": "integer"}},
                    "free_wifi_included": {"type": "boolean"},
                    "breakfast_included": {"type": "boolean"},
                    "gym_present": {"type": "boolean"},
                    "pool_present": {"type": "boolean"},
                    "is_pet_friendly": {"type": "boolean"},
                    "has_spa_services": {"type": "boolean"},
                    "smoking_allowed": {"type": "boolean"},
                    "is_wheelchair_accessible": {"type": "boolean"},
                    "has_free_parking": {"type": "boolean"},
                    "airport_shuttle_present": {"type": "boolean"},
                },
                "required": ["city", "checkin_date", "checkout_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_hotels",
            "description": "Filter previously searched hotel results by additional criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prior_results": {
                        "type": "string",
                        "description": "Variable name of prior results",
                    },
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "integer"},
                    "rating": {"type": "array", "items": {"type": "number"}},
                    "stars": {"type": "array", "items": {"type": "integer"}},
                    "free_wifi_included": {"type": "boolean"},
                    "breakfast_included": {"type": "boolean"},
                    "gym_present": {"type": "boolean"},
                    "pool_present": {"type": "boolean"},
                    "is_pet_friendly": {"type": "boolean"},
                },
                "required": ["prior_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for flights. Requires departure_date and origin/destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_airport_city": {"type": "string"},
                    "end_airport_city": {"type": "string"},
                    "departure_date": {"type": "array", "items": {"type": "string"}},
                    "arrival_date": {"type": "array", "items": {"type": "string"}},
                    "airline": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "integer"},
                    "flight_class": {"type": "array", "items": {"type": "string"}},
                    "num_layovers": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["departure_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_flights",
            "description": "Filter previously searched flight results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prior_results": {"type": "string"},
                    "airline": {"type": "array", "items": {"type": "string"}},
                    "budget": {"type": "integer"},
                    "flight_class": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prior_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search for restaurants in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "cuisine": {"type": "array", "items": {"type": "string"}},
                    "rating": {"type": "array", "items": {"type": "number"}},
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                    "price_range": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_restaurants",
            "description": "Filter previously searched restaurant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prior_results": {"type": "string"},
                    "cuisine": {"type": "array", "items": {"type": "string"}},
                    "rating": {"type": "array", "items": {"type": "number"}},
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prior_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_attractions",
            "description": "Search for attractions in a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "type": {"type": "array", "items": {"type": "string"}},
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_attractions",
            "description": "Filter previously searched attraction results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prior_results": {"type": "string"},
                    "type": {"type": "array", "items": {"type": "string"}},
                    "neighborhood": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prior_results"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_to_cache",
            "description": "Save results to cache with a unique key for later retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Unique cache key"},
                    "value": {
                        "type": "string",
                        "description": "Variable name of results to cache",
                    },
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_results_from_cache",
            "description": "Retrieve previously cached results by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Cache key to retrieve"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sort_results",
            "description": "Sort results by a specific field.",
            "parameters": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "string",
                        "description": "Variable name of results to sort",
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Field to sort by (e.g. price, rating)",
                    },
                    "ascending": {"type": "boolean"},
                },
                "required": ["results", "sort_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "seek_information",
            "description": "Ask the user for missing mandatory information before calling a tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask the user",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_date",
            "description": "Adjust a date by a number of days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["date", "days"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_nearest",
            "description": "Find nearest locations between two sets of results (e.g. hotels near restaurants).",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotels": {
                        "type": "string",
                        "description": "Variable name of hotel results",
                    },
                    "restaurants": {
                        "type": "string",
                        "description": "Variable name of restaurant results",
                    },
                    "attractions": {
                        "type": "string",
                        "description": "Variable name of attraction results",
                    },
                    "groupBy": {
                        "type": "string",
                        "description": "Group results by this entity type",
                    },
                },
                "required": [],
            },
        },
    },
]
