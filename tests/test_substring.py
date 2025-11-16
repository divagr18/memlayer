"""Debug with print statements."""

# Test the substring logic directly
name_lower = "dr. emma watson"
existing_lower = "dr. watson"

print(f"name_lower: '{name_lower}'")
print(f"existing_lower: '{existing_lower}'")
print(f"existing_lower in name_lower: {existing_lower in name_lower}")

# The issue is the substring matching is too simple
# "dr. watson" is NOT a substring of "dr. emma watson" because of the space!
# "dr." and "watson" are separated by "emma"
