
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import json, requests
import pandas as pd
import datetime

from dram_model import Fab_DRAM
from hdd_model  import Fab_HDD
from ssd_model  import Fab_SSD
from logic_model  import Fab_Logic

debug = False

avg_carbon_intensity = 0

# emaps key for the API
my_key = "" # add your electricity maps API
headers = {"auth-token": my_key}

EMAPS_regions = [
    "US-MIDW-MISO",
    "US-CENT-SWPP",
    "US-NW-PACW",
    "US-CAR-DUK",
    "US-SE-SOCO",
    "US-TEN-TVA",
    "US-MIDA-PJM",
    "US-NW-PACE",
    "US-TEX-ERCO",
    "US-CAL-CISO",
    "US-SW-PNM",
    "US-SW-SRP",
    "IE",  # ireland
    "SE",  # sweden
    "DK-DK1",  # west denmark
    "SG",  # singapore
    "NO-NO2",  # southwest norway
]

def embodied_carbon_estimate(area, hbm):

    
    chip_area = area/100000000 ## TODO convert from um^2 to cm^2  phaze uses (whatever accelergy uses)

    hbm = hbm # GB

    ###################
    # Keeping as constants
    ic_yield = 0.875
    node_technology = 22
    ###################


    chip_logic = Fab_Logic(gpa  = "95",
                      carbon_intensity = "loc_taiwan",
                      process_node = node_technology,
                      fab_yield=ic_yield)


    HBM = Fab_DRAM(config = "lpddr3_20nm", fab_yield = ic_yield)

    chip_logic.set_area(chip_area)
    HBM.set_capacity(hbm)

    HBM_co2 = HBM.get_carbon() / 1000. 
    chip_co2   = chip_logic.get_carbon()  / 1000.

    print("ACT HBM", HBM_co2, "kg CO2")
    print("ACT CHIP", chip_co2, "kg CO2")

    return chip_co2, HBM_co2

def operational_carbon_estimate(energy):
     # convert energy from Joules to kwh
     energy_kwh = energy / 3600000
     # avg_carbon_intensity is in gCO2 per kWh , convert to kgCO2
     return energy_kwh * avg_carbon_intensity / 1000 
     
def initialize_carbon_intensity(zone="US-CAL-CISO", start="2023-01-01", end="2024-01-01"):
    global avg_carbon_intensity

    # data = query_emaps_zone(zone, start, end)
    # avg_carbon_intensity = (data["AER"].mean())
    avg_carbon_intensity = 262.66632420091327 # use pre computed number
    print("Carbon Intensity initialized for " + zone + " in between " + start + " and " + end + " as: " + str(avg_carbon_intensity))

# Function to retrieve dara and format
def query_emaps_zone(zone, start_date, end_date):
    date = pd.to_datetime(start_date)
    date_end = pd.to_datetime(end_date)
    all_data = pd.DataFrame(columns=["ElectricityMaps Zone", "AER", "UTC_Timestamp"])

    # Emaps only allows data up to 10 days, so we quercy and average for an entire year
    max_days = 10
    while date < date_end:
        date_next = date + datetime.timedelta(max_days)
        #print(date.strftime("%Y-%m-%d"), ", ", date_next.strftime("%Y-%m-%d"))
        if(date_next > date_end):
            date_next = date_end
        url = "https://api.electricitymap.org/v3/carbon-intensity/past-range"
        querystring = {"zone":zone,"start":date.strftime("%Y-%m-%d") + "T00:00:00Z", "end":date_next.strftime("%Y-%m-%d") + "T00:00:00Z"}
        data = query_emaps(url, querystring)
        data = pd.DataFrame(data['data'])
        data = data.drop(columns=['updatedAt', 'createdAt', 'emissionFactorType', 'isEstimated', 'estimationMethod'])
        data = data.rename(columns={"zone":'ElectricityMaps Zone', "carbonIntensity":"AER", "datetime":"UTC_Timestamp"})
        all_data = pd.concat([all_data, data], ignore_index=True)
        date = date_next
    return all_data

def query_emaps(url, querystring):
    response = requests.get(url, params=querystring, headers=headers)
    if response.status_code == requests.codes.ok:
            # Parse JSON to get a pandas.DataFrame of data and dict of metadata
            parsed_response = json.loads(response.text)
            return parsed_response
    else:
        print("Request is not successful: " + response.text)
        return


