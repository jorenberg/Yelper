#!/usr/bin/env python


__authors__ 	= [
    '"Prabhat Kumar" <prabhat.genome@gmail.com>',
    '"Aiswarya Thomas" <aiswaryathomas15@gmail.com>',
    '"Sequømics Corporation" <admin@sequomics.com>'
    ]
__company__ 	= 'Sequømics Corporation'
__homepage__ 	= 'http://sequomics.com/'
__account__ 	= 'SequomicsCorporation'
__githubURL__   = 'https://github.com/SequomicsCorporation'
__license__     = 'Apache License'

# ------------------------------------------------------------------------
# Copyright © 2015, Sequømics Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       	http://www.apache.org/licenses/LICENSE-2.0
#                           	or
#   https://github.com/SequomicsCorporation/Yelper/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import csv
import json
import nltk
import numpy as np
# ========================================================================
# Python’s class mechanism [ETL]: B: /yelper/etl/
# ========================================================================
import random
from pandas import DataFrame

class ETLUtils:
    # Defining Function and __init__ method as a constructor [private].
    def __init__(self):
        pass
    
    
    # @staticmethod declarations.
    # A way to write a method inside a class without reference to the object it is being called on.
    @staticmethod
    def load_json_file(file_path):
        """
        Builds a list of dictionaries from a JSON file
            :type file_path: string
            :param file_path: the path for the file that contains the businesses data.
            :return: a list of dictionaries with the data from the files
        """
        records = [json.loads(line) for line in open(file_path)]
        return records
    
    @staticmethod
    def save_json_file(file_path, records):
        with open(file_path, 'w') as outfile:
            for record in records:
                json.dump(record, outfile)
                outfile.write('\n')
    
    @staticmethod
    def drop_fields(fields, dictionary_list):
        """
        Removes the specified fields from every dictionary in the dictionary list
            :rtype : void
            :param fields: a list of strings, which contains the fields that are
             going to be removed from every dictionary in the dictionary list
            :param dictionary_list: a list of dictionaries
        """
        for dictionary in dictionary_list:
            for field in fields:
                del (dictionary[field])
