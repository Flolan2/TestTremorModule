#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:01:47 2024

@author: Lange_L
"""

# data_loader.py

import json

def load_json(json_path):
    """Load and parse the JSON file containing tracking data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
