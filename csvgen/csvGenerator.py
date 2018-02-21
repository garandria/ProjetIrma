#!/usr/bin/python3

<<<<<<< HEAD
#   Copyright 2018 TuxML Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

=======
>>>>>>> 48a1dddd0a0c8d66d7b16ba305d3ef3a8c338f1a
import os
import re

config_folder = "configs/"

config_files = os.listdir(config_folder)

file_count = len(config_files)

properties = {"KERNEL_SIZE":[0]*file_count}

pattern = re.compile("^([^#][^=]*)=(.*)$")

for file_number, config_file in enumerate(config_files):
    config_content = open(config_folder + config_file, "r")
    # Config properties
    for line in config_content:
        m = re.match(pattern, line)
        if m:
            key = m.group(1)
            value = m.group(2)
            if key not in properties:
                properties[key] = ["n"]*file_count
            properties[key][file_number] = value
    # File size
    properties["KERNEL_SIZE"][file_number] = str(os.path.getsize(config_folder + config_file))

keys_line = ""
value_lines = [""]*file_count
for (k,v) in properties.items():
    keys_line += k + ","
    for i in range(file_count):
        value_lines[i] += v[i] + ","

print(keys_line)
for line in value_lines:
    print(line)

