import json
from collections import Counter

def analyze_json(filename='all_formulas.json'):
    """
    Reads and analyzes the structure of a JSON file, providing a detailed
    description and counts of its contents.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: The file '{filename}' was not found in this directory.")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Could not parse '{filename}'. The file may not be valid JSON.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print(f"--- 📊 Analysis of {filename} ---")

    top_level_type = type(data).__name__

    # Case 1: The JSON's top level is a LIST (e.g., `[...]`)
    if isinstance(data, list):
        count = len(data)
        print(f"\n📈 **Top-Level Structure:** A **list** containing **{count}** elements.")

        if count > 0:
            # Analyze the types of items in the list
            element_types = Counter(type(item).__name__ for item in data)
            print(f"\n🔍 **Element Types Found:**")
            for item_type, num in element_types.items():
                print(f"  - `{item_type}`: {num} occurrence(s)")

            # If dictionaries are the primary type, analyze their keys
            if 'dict' in element_types:
                dict_items = [item for item in data if isinstance(item, dict)]
                # Find all unique sets of keys and count them
                key_structures = Counter(tuple(sorted(item.keys())) for item in dict_items)
                
                print(f"\n🔑 **Dictionary Key Structures:**")
                if not key_structures:
                     print("  - No dictionaries found to analyze keys.")
                else:
                    for i, (keys, num) in enumerate(key_structures.items()):
                        print(f"  - **Structure #{i+1}** ({num} occurrences):")
                        print(f"    - Keys: {list(keys)}")

    # Case 2: The JSON's top level is a DICTIONARY (e.g., `{...}`)
    elif isinstance(data, dict):
        count = len(data)
        print(f"\n📈 **Top-Level Structure:** A **dictionary** (object) with **{count}** key-value pairs.")
        print(f"\n🔑 **Keys:** {list(data.keys())}")

    # Case 3: The JSON contains a single value (e.g., a number or string)
    else:
        print(f"The file contains a single value of type **{top_level_type}**.")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    analyze_json()
