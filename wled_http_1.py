# This program is a test to verify HTTP communication
# between a Raspberry Pi and an ESP32 running WLED

import requests

# Replace with your WLED device's IP address
wled_ip = 'http://192.168.1.146'

# Function to set effect and color palette
def set_wled_effect_and_palette(effect_id, palette_id):
    url = f'{wled_ip}/win&FX={effect_id}&FP={palette_id}'
    response = requests.get(url)
    if response.status_code == 200:
        print(f'Successfully set effect {effect_id} and palette {palette_id}')
    else:
        print(f'Failed to set effect and palette. Status code: {response.status_code}')

# Continuous loop for testing
try:
    print("Press Ctrl+C to exit.")
    while True:
        try:
            effect_id = int(input('Enter the effect ID (0-101): '))
            palette_id = int(input('Enter the palette ID (0-55): '))
            set_wled_effect_and_palette(effect_id, palette_id)
        except ValueError:
            print("Invalid input. Please enter numeric values for effect and palette IDs.")
except KeyboardInterrupt:
    print("\nExiting the script.")
