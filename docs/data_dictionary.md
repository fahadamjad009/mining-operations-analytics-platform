Data Dictionary



This project supports two data modes: Mining (synthetic) and NASA C-MAPSS (public benchmark).

Both are transformed into analytics-ready tables for modelling and reporting.



--------------------------------

RAW DATA - MINING MODE

--------------------------------



telemetry.parquet

ts                Timestamp of telemetry reading

site              Mining site name

equipment\_id      Unique asset identifier

equipment\_type    haul\_truck, excavator, drill

shift             day or night

engine\_temp\_c     Engine temperature (Celsius)

vibration\_mm\_s    Vibration level

fuel\_rate\_lph     Fuel consumption (litres per hour)

speed\_kmh         Operating speed

event             running, idle, or down



maintenance.parquet

ts                Timestamp of maintenance event

site              Mining site

equipment\_id      Asset identifier

work\_order\_id     Maintenance job ID

fault\_code        Type of fault

severity          low, medium, high

downtime\_minutes  Minutes asset was unavailable



--------------------------------

PROCESSED DATA (COMMON TO BOTH MODES)

--------------------------------



daily\_equipment\_summary.parquet

site

equipment\_id

equipment\_type

date

points             Number of telemetry records for the day

running            Count of running states

idle               Count of idle states

down               Count of down states

avg\_temp           Average engine temperature

avg\_vib            Average vibration

avg\_fuel           Average fuel rate

utilization\_rate   running / points

downtime\_rate      down / points

work\_orders        Number of maintenance jobs

downtime\_minutes   Total downtime minutes

high\_sev           Count of high severity faults



--------------------------------

MODEL FEATURES

--------------------------------



avg\_temp

avg\_vib

avg\_fuel

utilization\_rate

downtime\_rate

work\_orders

high\_sev



--------------------------------

MODEL TARGET

--------------------------------



label\_high\_downtime

1 = asset has >= 120 minutes downtime next day

0 = asset has < 120 minutes downtime next day



