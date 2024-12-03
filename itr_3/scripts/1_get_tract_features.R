# Author: Ashlynn Wimer
# Date: 11/29/2024
# About: Get and clean census data for this study!
# We use R for this not because we want to, but because
# it seems that most of the Census data packages in 
# Python (including my own) are either broken or Very
# Glitchy. 

library(tidycensus)
library(dplyr)
library(tidyr)
library(sf)


## Read in city shapefiles
seattle <- st_read('../data/shapes/seattle_boundaries.shp')
new_york <- st_read('../data/shapes/ny_boundaries.shp')
cincinnati <- st_read('../data/shapes/cinc_boundaries.shp')

## Read in crashes
seattle_crashes <- st_read('../data/shapes/seattle_crashes.csv',
  options=c('X_POSSIBLE_NAMES=x', 'Y_POSSIBLE_NAMES=y')) |>
  st_set_crs("EPSG:4236")
new_york_crashes <- st_read('../data/shapes/ny_crashes.shp')
cinc_crashes <- st_read('../data/shapes/cincinnati_crashes.csv', 
  options=c("X_POSSIBLE_NAMES=LATITUDE_X", "Y_POSSIBLE_NAMES=LONGITUDE_X"))  |>
  st_set_crs("EPSG:4236")

# Given a state and crash point data
# Make a dataframe of crash point data counts
get_crash_counts <- function(crashes, state) {
  tracts <- tigris::tracts(state=state) |>
    select(GEOID) |>
    st_transform(st_crs(crashes))
  
  (tracts$numCrashes <- lengths(st_intersects(tracts, crashes)))
  
  tracts <- select(tracts, GEOID, numCrashes) |> st_drop_geometry()
  return(tracts)
}

# Get Census variables for our study
# We steal variable selection from
# Apardian and Smirnov (2020)
get_variables <- function(city_shapefile, crashes, state) {
  
  ### Load and Clean variables
  # ACS
  acs <- tidycensus::get_acs(
      variables=c(
          "medHouseIncome"="B19049_001",       "totalPop"="B01003_001",
          "total_black_pop" ="B02001_003",     "medHouseVal"="B25077_001",
          "less_hs"="B16010_002" ,             "some_college"="B16010_028",
          "bach_plus"="B16010_041",            "labor_force"="C18120_002",
          "unempl"="C18120_006" ,              "under_5"="B06001_002",
          "5_to_17"="B06001_003" ,             "65_to_74"="B06001_011",
          "75_plus"="B06001_012" ,             "agg_work_travel"="B08135_001",
          "ed_universe" = 'B16010_001',        "health_universe"="B992701_001",
          "without_insurance"="B992701_003"
      ), year=2022, geography='tract', state=state) |>
    select(GEOID, variable, estimate) |>
    pivot_wider(names_from='variable', values_from='estimate') |>
    mutate(
      perBlack = round(100 * total_black_pop / totalPop, 2),
      unemplRate = round(100 * unempl / labor_force, 2),
      perUnder5 = round(100 * under_5 / totalPop, 2),
      per5to17  = round(100 * `5_to_17` / totalPop, 2),
      perOver65 = round(100 * (`65_to_74` + `75_plus`) / totalPop),
      percentNoHs = round(100 * less_hs / ed_universe, 2),
      percentBach = round(100 * bach_plus / ed_universe, 2),
      percentSomeCol = round(100 * some_college / ed_universe, 2),
      workTravelAv = round(agg_work_travel / labor_force, 2),
      perWithoutIns = round(without_insurance / health_universe, 2)
    ) |>
    select(-total_black_pop, -labor_force,
           -under_5, -`5_to_17`, -`65_to_74`, -`75_plus`,
           -ed_universe, -agg_work_travel, -health_universe, 
           -without_insurance, -less_hs, -bach_plus, -some_college)
  # DHC
  dhc <- tidycensus::get_decennial(geography='tract', 
                                   variables=c("owner_occ"="H10_002N",
                                               "totHousing"='H10_001N'),
                                   year=2020, sumfile = 'dhc', state=state) |>
    select(GEOID, variable, value) |>
    pivot_wider(names_from='variable', values_from='value') |>
    mutate(
      perOwnerOcc = round(100 * owner_occ / totHousing, 2)
    ) |>
    select(-owner_occ)
  
  # Geometries
  tracts <- tigris::tracts(state=state) |>
    select(GEOID, ALAND)
  
  # Crashes
  crash_counts <- get_crash_counts(crashes, state)
  
  ## Merge
  tracts <- tracts |> 
    merge(dhc, by='GEOID') |> 
    merge(acs, by='GEOID') |>
    merge(crash_counts, by='GEOID') |>
    mutate(popDens = round(ALAND / totalPop, 2)) |>
    select(-ALAND)
  
  ## Spatial Filter
  old_crs <- st_crs(tracts)
  tracts <- tracts |> 
    st_transform(st_crs(city_shapefile)) |>
    st_filter(city_shapefile) |> 
    st_transform(old_crs)
  
  return(tracts)
}

merge_redlining <- function(df, redlining_loc='../data/shapes/redlining/HRI2010.shp') {
  redline_tracts <- st_read(redlining_loc) |>
    select(GEOID=GEOID10, HRI2010) |>
    st_drop_geometry()

  return(merge(df, redline_tracts, by='GEOID', all.x=TRUE))
}

## Get city census data + merge in redlining data
seattle_data <- get_variables(seattle, seattle_crashes, 'Washington') |>
  merge_redlining()
nyc_data <- get_variables(new_york, new_york_crashes, 'New York') |>
  merge_redlining()
cincinnati_data <- get_variables(cincinnati, cinc_crashes, 'Ohio') |>
  merge_redlining()

## Save
st_write(seattle_data, "../data/shapes/seattle_census.gpkg", append=FALSE)
st_write(nyc_data, "../data/shapes/nyc_census.gpkg", append=FALSE)
st_write(cincinnati_data, '../data/shapes/cinc_census.gpkg', append=FALSE)