def openai_config(model):
    if model == 'gpt-35-turbo':
        config = {
            "model": "gpt-35-turbo",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
            "api_version": "<YOUR_API_VERSION>",
            # "api_type": "AZURE"
        }
    elif model == 'kimi':
        config = {
            "model": "moonshot-v1-8k",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
        }
    elif model == 'gpt-4o-mini-2024-07-18':
        config = {
            "model": "gpt-4o-mini-2024-07-18",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
        }    
    elif model == 'gpt-4.1-nano':
        config = {
            "model": "gpt-4.1-nano",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
        } 

    elif model == 'o1-plus':
        config = {
            "model": "o1-plus",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
        } 

    elif model == 'gpt-3.5-turbo-0125':
        config = {
            "model": "gpt-3.5-turbo-0125",
            "api_key": "<YOUR_API_KEY>",
            "base_url": "<YOUR_BASE_URL>",
        } 
    return config

def llm_config_list(seed, config_list):
    llm_config_list = {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 240,
        "cache_seed": seed,
        "temperature": 0,
    }
    return llm_config_list