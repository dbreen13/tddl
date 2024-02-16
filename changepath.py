# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

directory = "/home/dbreen/Documents/tddl/papers/iclr_2023/configs/garipov/fmnist/decompose"

for filename in os.listdir(directory):
    if filename.startswith("dec-"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            file_content = file.read()
        
        # Replace the specified part of the path
        modified_content = file_content.replace("/home/demi/", "/home/dbreen/")
        
        # Write the modified content back to the file
        with open(filepath, 'w') as file:
            file.write(modified_content)
