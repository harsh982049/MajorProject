import pandas as pd

# Step 1: Load the datasets
keystrokes_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\keystrokes.tsv", sep='\t')
usercondition_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\usercondition.tsv", sep='\t')
mouse_mov_speeds_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\mouse_mov_speeds.tsv", sep='\t')
mousedata_df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\ML\\mousedata.tsv", sep='\t')

# Drop irrelevant/unnamed columns (if they exist)
keystrokes_df = keystrokes_df.drop(columns=['Unnamed: 4'], errors='ignore')
usercondition_df = usercondition_df.drop(columns=['Unnamed: 7'], errors='ignore')
mouse_mov_speeds_df = mouse_mov_speeds_df.drop(columns=['Unnamed: 3'], errors='ignore')
mousedata_df = mousedata_df.drop(columns=['Unnamed: 5'], errors='ignore')

# Convert all 'Time' columns to datetime
keystrokes_df['Press_Time'] = pd.to_datetime(keystrokes_df['Press_Time'], errors='coerce')
keystrokes_df['Relase_Time'] = pd.to_datetime(keystrokes_df['Relase_Time'], errors='coerce')
usercondition_df['Time'] = pd.to_datetime(usercondition_df['Time'], errors='coerce')
mouse_mov_speeds_df['Time'] = pd.to_datetime(mouse_mov_speeds_df['Time'], errors='coerce')
mousedata_df['Time'] = pd.to_datetime(mousedata_df['Time'], errors='coerce')

# Drop rows with any invalid timestamps
keystrokes_df = keystrokes_df.dropna(subset=['Press_Time', 'Relase_Time'])
usercondition_df = usercondition_df.dropna(subset=['Time'])
mouse_mov_speeds_df = mouse_mov_speeds_df.dropna(subset=['Time'])
mousedata_df = mousedata_df.dropna(subset=['Time'])

# Sort all datasets by time
keystrokes_df = keystrokes_df.sort_values(by='Press_Time')
usercondition_df = usercondition_df.sort_values(by='Time')
mouse_mov_speeds_df = mouse_mov_speeds_df.sort_values(by='Time')
mousedata_df = mousedata_df.sort_values(by='Time')

# Print the first few rows of each for confirmation
print("Cleaned Keystrokes:")
print(keystrokes_df.head(), "\n")

print("Cleaned User Conditions:")
print(usercondition_df.head(), "\n")

print("Cleaned Mouse Speeds:")
print(mouse_mov_speeds_df.head(), "\n")

print("Cleaned Mouse Events:")
print(mousedata_df.head(), "\n")