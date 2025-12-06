# Violation Type Settings Guide

## Overview

The system now allows you to configure which clothing items are considered violations through the Settings page. This makes the system flexible and adaptable to different dress code requirements.

## How to Use

### 1. Access Settings
- Navigate to the **Settings** page in the Streamlit dashboard
- Scroll down to the **🚫 Violation Type Configuration** section

### 2. Select Violation Types
- You'll see a multi-select dropdown with all available clothing items
- Select the items that should trigger a violation when detected
- Click **💾 Save Violation Settings** to apply your changes

### 3. Reset to Defaults
- Click **🔄 Reset to Defaults** to restore the original violation types:
  - shorts
  - skirt
  - flipflops
  - sandals
  - vest
  - sling_dress
  - sling

### 4. View Current Settings
- The current violation types are displayed below the selection area
- Shows how many violation types are configured

## How It Works

1. **Settings Storage**: Violation types are saved to `model/violation_settings.json`
2. **Dynamic Loading**: The system loads violation settings each time clothing detection runs
3. **Immediate Effect**: Changes take effect immediately - no restart required
4. **Works Everywhere**: Settings apply to both CLI (`main.py`) and Dashboard

## Available Clothing Items

The system includes a comprehensive list of clothing items that can be detected:

**Tops:**
- short_sleeve_top
- long_sleeve_top
- short_sleeve_outwear
- long_sleeve_outwear
- vest
- sling
- crop_top
- tank_top
- sleeveless

**Bottoms:**
- shorts
- trousers
- skirt

**Dresses:**
- short_sleeve_dress
- long_sleeve_dress
- vest_dress
- sling_dress

**Footwear:**
- shoes
- sandals
- flipflops
- sneakers
- boots

**Accessories:**
- hat
- cap

## Technical Details

### Files Modified/Created

1. **`py/violation_settings.py`** (NEW)
   - Manages loading/saving violation settings
   - Provides default values
   - Handles settings persistence

2. **`py/clothing_analysis.py`** (MODIFIED)
   - Now loads violation settings dynamically
   - Uses user-selected violation types instead of hardcoded list
   - `is_banned()` function accepts violation types as parameter

3. **`frontend/pages/3_Settings.py`** (MODIFIED)
   - Added violation type configuration UI
   - Multi-select dropdown for selecting violations
   - Save and reset buttons

### Settings File Location

Violation settings are stored in:
```
model/violation_settings.json
```

Example content:
```json
{
  "violation_types": [
    "shorts",
    "skirt",
    "flipflops",
    "sandals"
  ]
}
```

## Usage Examples

### Example 1: Strict Dress Code
Select all casual items as violations:
- shorts
- flipflops
- sandals
- tank_top
- sleeveless

### Example 2: Relaxed Dress Code
Only select obvious violations:
- flipflops
- sandals

### Example 3: Custom Requirements
Select specific items based on your organization's policy:
- Any combination of items from the available list

## Notes

- **No Restart Required**: Changes take effect immediately when you save
- **Persistent**: Settings are saved to disk and persist across sessions
- **Backward Compatible**: If no settings file exists, defaults are used
- **CLI Compatible**: Works with both Streamlit dashboard and command-line interface

## Troubleshooting

**Settings not saving?**
- Check that the `model/` directory exists and is writable
- Verify you clicked "Save Violation Settings" button

**Changes not taking effect?**
- Settings are loaded dynamically, so they should work immediately
- If using CLI, make sure you're using the updated `clothing_analysis.py`

**Want to add more clothing items?**
- Edit `py/violation_settings.py`
- Add items to the `ALL_AVAILABLE_ITEMS` list
- Restart the application

