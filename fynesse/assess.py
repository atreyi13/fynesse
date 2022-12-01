from .config import *

from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""
import pandas as pd
import osmnx as ox
import numpy as np
import mlai
import mlai.plot as plot
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import math
from scipy.spatial import distance_matrix

def data(county, datestart,dateend,country, conn, city = None): # This gets the data from access, No need to run this if access step has been excecuted
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    access.create_price_coord_data(conn)
    df = pd.DataFrame()
    if city:
        df= access.access_city(county, city, datestart,dateend,country, conn) 
    else:
        df= access.access_county(county, datestart,dateend,country, conn) 
    return df.dropna()    
        
def get_pois(df): #This returns pois given a dataframe
    north = df['lattitude'].max() + 0.02
    south = df['lattitude'].min() - 0.02
    west = df['longitude'].min() - 0.02
    east = df['longitude'].max() + 0.02

    tags = {"amenity": True, 
            "buildings": True, 
            "historic": True, 
            "leisure": True, 
            "shop": True, 
            "tourism": True}
    pois = ox.geometries_from_bbox(north, south, east, west, tags)
    return pois        

def get_geo_graph(df,place_name): #This returns nodes, edges and area from coordinates. No need to run this separately. A call to this function is in plot_geo_graph below.
    north = df['lattitude'].max() + 0.02
    south = df['lattitude'].min() - 0.02
    west = df['longitude'].min() - 0.02
    east = df['longitude'].max() + 0.02
    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)
    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(place_name)
    return nodes, edges, area

def plot_geo_graph(df,place_name, pois): #We return a graph of the pois in this function
    north = df['lattitude'].max() + 0.02
    south = df['lattitude'].min() - 0.02
    west = df['longitude'].min() - 0.02
    east = df['longitude'].max() + 0.02
    
    nodes, edges, area = get_geo_graph(df, place_name)   
    fig, ax = plt.subplots(figsize=plot.big_figsize)

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs 
    pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    plt.tight_layout()
    mlai.write_figure(directory="./maps", filename="cambridge-england-pois.svg")
    return fig

def plot_log_prices_over_region(df): #This returns a graph that plots the log distribution of house prices over the given region. Colorbar maps colour to house prices.
    z = df['price']
    x = df['longitude'] 
    y = df['lattitude']
    c = np.log(z)

    fig1, ax1 = plt.subplots(figsize=(7,9))
    ax2 = plt.subplot()
    ax1.scatter(x, y, c=c, cmap='rainbow', alpha=0.2)
    im = ax2.scatter(x, y, c=c, cmap='rainbow', alpha=1.0)
    cax=fig1.add_axes([0.95,0.12,0.05,0.75]) 
    fig1.text(0.5,0.07, "Longitude", ha="center", va="center")
    fig1.text(0.05,0.5, "Latitude", ha="center", va="center", rotation=90)
    fig1.text(0.5,0.01, "Prices over region", ha="center", va="center")
    cb = fig1.colorbar(im, ax=ax1, orientation="vertical", shrink=0.9, cax = cax)

    def label_exp(x,pos):
        return "{:4.2f}".format(np.exp(x))

    cb.formatter = ticker.FuncFormatter(label_exp)
    cb.update_ticks()
    plt.savefig('House_price_distribution_2010.png', bbox_inches = 'tight')
    return fig1

def plot_prices_over_region(df):  #This returns a graph that plots the distribution of house prices over the given region. Colorbar maps colour to house prices.
    z = df['price']
    x = df['longitude'] 
    y = df['lattitude']
    c = z

    fig1, ax1 = plt.subplots(figsize=(7,9))
    ax2 = plt.subplot()
    ax1.scatter(x, y, c=c, cmap='rainbow', alpha=0.2)
    im = ax2.scatter(x, y, c=c, cmap='rainbow', alpha=1.0)
    cax=fig1.add_axes([0.95,0.12,0.05,0.75]) 
    fig1.text(0.5,0.07, "Longitude", ha="center", va="center")
    fig1.text(0.05,0.5, "Latitude", ha="center", va="center", rotation=90)
    fig1.text(0.5,0.01, "Prices over region", ha="center", va="center")

    cb = fig1.colorbar(im, ax=ax2, orientation="vertical", shrink=0.9, cax = cax)

    plt.savefig('House_price_distribution_2010.png', bbox_inches = 'tight')
    return fig1

def prices_over_time(df): #This returns a graph of the prices over time of different types of houses. The bounds are the max and min house prices.
    y1 = (df[df['property_type'] == 'D'].groupby(pd.PeriodIndex(df[df['property_type'] == 'D']['date_of_transfer'], freq="M"))['price'])
    y2 = (df[df['property_type'] == 'F'].groupby(pd.PeriodIndex(df[df['property_type'] == 'F']['date_of_transfer'], freq="M"))['price'])
    y3 = (df[df['property_type'] == 'O'].groupby(pd.PeriodIndex(df[df['property_type'] == 'O']['date_of_transfer'], freq="M"))['price'])
    y4 = (df[df['property_type'] == 'S'].groupby(pd.PeriodIndex(df[df['property_type'] == 'S']['date_of_transfer'], freq="M"))['price'])
    y5 = (df[df['property_type'] == 'T'].groupby(pd.PeriodIndex(df[df['property_type'] == 'T']['date_of_transfer'], freq="M"))['price'])

    x = sorted(list(set((pd.DatetimeIndex( df['date_of_transfer']).to_period('M')).strftime('%Y%m'))))
    x3 = sorted(list(set((pd.DatetimeIndex( df[df['property_type'] == 'O']['date_of_transfer']).to_period('M')).strftime('%Y%m'))))
    newticks = [j for i,j in enumerate(x) if not i%24]
    #newticks = x

    fig = plt.figure(figsize=(15,15))
    (ax1, ax2, ax3) = fig.subplots(3, 2)
    ax1[0].plot(x, y1.mean(), color='red', linestyle='--', zorder=1, label="Detached")
    ax1[0].plot(x, y2.mean(), color='cyan', linestyle='--', zorder=1, label="Flats/Maisonettes")
    ax1[0].plot(x3, y3.mean(), color='green', linestyle='--', zorder=1, label="Other")
    ax1[0].plot(x, y4.mean(), color='orange', linestyle='--', zorder=1, label="Semi-Detached")
    ax1[0].plot(x, y5.mean(), color='magenta', linestyle='--', zorder=1, label="Terraced")
    ax1[0].legend()
    ax1[0].set_xticks(newticks)
    ax1[0].set_xticklabels(newticks, rotation = 45)
    ax1[0].set_xlabel('months') 
    ax1[0].set_ylabel('price') 

    ax1[1].plot(x, y1.mean(), color='red', linestyle='--', zorder=1)
    ax1[1].plot(x, y2.mean(), color='cyan', linestyle='--', zorder=1)
    ax1[1].plot(x, y4.mean(), color='orange', linestyle='--', zorder=1)
    ax1[1].plot(x, y5.mean(), color='magenta', linestyle='--', zorder=1)
    ax1[1].set_xticks(newticks)
    ax1[1].set_xticklabels(newticks, rotation = 45)
    ax1[1].set_xlabel('months') 
    ax1[1].set_ylabel('price') 

    ax2[0].plot(x, y1.max(),color='red',linestyle='-',zorder=1)
    ax2[0].plot(x, y1.mean(), color='red', linestyle='--', zorder=1)
    ax2[0].plot(x, y1.min(),color='red',linestyle='-',zorder=1)
    ax2[0].fill_between(x, y1.min(),y1.max(), color='red', alpha=.3)
    ax2[0].set_xticks(newticks)
    ax2[0].set_xticklabels(newticks, rotation = 45)
    ax2[0].set_xlabel('months') 
    ax2[0].set_ylabel('price') 

    ax2[1].plot(x, y2.max(),color='cyan',linestyle='-',zorder=1)
    ax2[1].plot(x, y2.mean(), color='cyan', linestyle='--', zorder=1)
    ax2[1].plot(x, y2.min(),color='cyan',linestyle='-',zorder=1)
    ax2[1].fill_between(x, y2.min(),y2.max(), color='cyan', alpha=.3)
    ax2[1].set_xticks(newticks)
    ax2[1].set_xticklabels(newticks, rotation = 45)
    ax2[1].set_xlabel('months') 
    ax2[1].set_ylabel('price') 

    ax3[0].plot(x, y4.max(),color='orange',linestyle='-',zorder=1)
    ax3[0].plot(x, y4.mean(), color='orange', linestyle='--', zorder=1)
    ax3[0].plot(x, y4.min(),color='orange',linestyle='-',zorder=1)
    ax3[0].fill_between(x, y4.min(),y4.max(), color='orange', alpha=.3)
    ax3[0].set_xticks(newticks)
    ax3[0].set_xticklabels(newticks, rotation = 45)
    ax3[0].set_xlabel('months') 
    ax3[0].set_ylabel('price')

    ax3[1].plot(x, y5.max(),color='magenta',linestyle='-',zorder=1)
    ax3[1].plot(x, y5.mean(), color='magenta', linestyle='--', zorder=1)
    ax3[1].plot(x, y5.min(),color='magenta',linestyle='-',zorder=1)
    ax3[1].fill_between(x, y5.min(),y5.max(), color='magenta', alpha=.3)
    ax3[1].set_xticks(newticks)
    ax3[1].set_xticklabels(newticks, rotation = 45)
    ax3[1].set_xlabel('months') 
    ax3[1].set_ylabel('price')

    fig.text(0.5,0.01, "Prices over time", ha="center", va="center")

    plt.tight_layout()
    plt.savefig('prices_overtime_2010.png', bbox_inches = 'tight')
    return fig

def log_price_over_time(df): #This is the same as the previous view, except weare taking log of the prices

    y1 = (df[df['property_type'] == 'D'].groupby(pd.PeriodIndex(df[df['property_type'] == 'D']['date_of_transfer'], freq="M"))['price'])
    y2 = (df[df['property_type'] == 'F'].groupby(pd.PeriodIndex(df[df['property_type'] == 'F']['date_of_transfer'], freq="M"))['price'])
    y3 = (df[df['property_type'] == 'O'].groupby(pd.PeriodIndex(df[df['property_type'] == 'O']['date_of_transfer'], freq="M"))['price'])
    y4 = (df[df['property_type'] == 'S'].groupby(pd.PeriodIndex(df[df['property_type'] == 'S']['date_of_transfer'], freq="M"))['price'])
    y5 = (df[df['property_type'] == 'T'].groupby(pd.PeriodIndex(df[df['property_type'] == 'T']['date_of_transfer'], freq="M"))['price'])

    x = sorted(list(set((pd.DatetimeIndex( df['date_of_transfer']).to_period('M')).strftime('%Y%m'))))
    x3 = sorted(list(set((pd.DatetimeIndex( df[df['property_type'] == 'O']['date_of_transfer']).to_period('M')).strftime('%Y%m'))))
    newticks = [j for i,j in enumerate(x) if not i%24]
    #newticks = x

    fig = plt.figure(figsize=(15,15))
    (ax1, ax2, ax3) = fig.subplots(3, 2)
    ax1[0].plot(x, np.log(y1.mean()), color='red', linestyle='--', zorder=1, label="Detached")
    ax1[0].plot(x, np.log(y2.mean()), color='cyan', linestyle='--', zorder=1, label="Flats/Maisonettes")
    ax1[0].plot(x3, np.log(y3.mean()), color='green', linestyle='--', zorder=1, label="Other")
    ax1[0].plot(x, np.log(y4.mean()), color='orange', linestyle='--', zorder=1, label="Semi-Detached")
    ax1[0].plot(x, np.log(y5.mean()), color='magenta', linestyle='--', zorder=1, label="Terraced")
    ax1[0].legend()
    ax1[0].set_xticks(newticks)
    ax1[0].set_xticklabels(newticks, rotation = 45)
    ax1[0].set_xlabel('months') 
    ax1[0].set_ylabel('price') 

    ax1[1].plot(x, np.log(y1.mean()), color='red', linestyle='--', zorder=1)
    ax1[1].plot(x, np.log(y2.mean()), color='cyan', linestyle='--', zorder=1)
    ax1[1].plot(x, np.log(y4.mean()), color='orange', linestyle='--', zorder=1)
    ax1[1].plot(x, np.log(y5.mean()), color='magenta', linestyle='--', zorder=1)
    ax1[1].set_xticks(newticks)
    ax1[1].set_xticklabels(newticks, rotation = 45)
    ax1[1].set_xlabel('months') 
    ax1[1].set_ylabel('price') 

    ax2[0].plot(x, np.log(y1.max()),color='red',linestyle='-',zorder=1)
    ax2[0].plot(x, np.log(y1.mean()), color='red', linestyle='--', zorder=1)
    ax2[0].plot(x, np.log(y1.min()),color='red',linestyle='-',zorder=1)
    ax2[0].fill_between(x, np.log(y1.min()),np.log(y1.max()), color='red', alpha=.3)
    ax2[0].set_xticks(newticks)
    ax2[0].set_xticklabels(newticks, rotation = 45)
    ax2[0].set_xlabel('months') 
    ax2[0].set_ylabel('price')

    ax2[1].plot(x, np.log(y2.max()),color='cyan',linestyle='-',zorder=1)
    ax2[1].plot(x, np.log(y2.mean()), color='cyan', linestyle='--', zorder=1)
    ax2[1].plot(x, np.log(y2.min()),color='cyan',linestyle='-',zorder=1)
    ax2[1].fill_between(x, np.log(y2.min()),np.log(y2.max()), color='cyan', alpha=.3)
    ax2[1].set_xticks(newticks)
    ax2[1].set_xticklabels(newticks, rotation = 45)
    ax2[1].set_xlabel('months') 
    ax2[1].set_ylabel('price')

    ax3[0].plot(x, np.log(y4.max()),color='orange',linestyle='-',zorder=1)
    ax3[0].plot(x,np.log(y4.mean()), color='orange', linestyle='--', zorder=1)
    ax3[0].plot(x, np.log(y4.min()),color='orange',linestyle='-',zorder=1)
    ax3[0].fill_between(x, np.log(y4.min()),np.log(y4.max()), color='orange', alpha=.3)
    ax3[0].set_xticks(newticks)
    ax3[0].set_xticklabels(newticks, rotation = 45)
    ax3[0].set_xlabel('months') 
    ax3[0].set_ylabel('price')

    ax3[1].plot(x, np.log(y5.max()),color='magenta',linestyle='-',zorder=1)
    ax3[1].plot(x, np.log(y5.mean()), color='magenta', linestyle='--', zorder=1)
    ax3[1].plot(x, np.log(y5.min()),color='magenta',linestyle='-',zorder=1)
    ax3[1].fill_between(x, np.log(y5.min()),np.log(y5.max()), color='magenta', alpha=.3)
    ax3[1].set_xticks(newticks)
    ax3[1].set_xticklabels(newticks, rotation = 45)
    ax3[1].set_xlabel('months') 
    ax3[1].set_ylabel('price')
    fig.text(0.5,0.01, "Prices over time", ha="center", va="center")

    plt.tight_layout()

    plt.savefig('prices_overtime_log_2010.png', bbox_inches = 'tight')
    return fig

def get_tags(pois): #This function simplifies the pois dataframe. It puts pois relevant to housing(like amenities, shops) in a single column. Each row also specifies the latitude and longitude of the poi. 
    #It is not necessary to run this to get training vectors since a call to this function is included in those functions.
    #This is the first point of feature reduction based on human bias.
  tags ={
      'amenity' : {'food':['bar', 'cafe', 'food_court','fast_food',  'ice_cream', 'pub', 'restaurant', 'biergarten', ],
                   'education' : ['college', 'driving_school', 'kindergarten', 'language_school', 'library', 'toy_library', 'training', 'music_school', 'school', 'university'],
                   'transportation' : ['bicycle_parking', 'bicycle_rental', 'boat_rental', 'bus_station', 'car_rental', 'car_wash', 'charging_station', 'fuel', 'parking', 'taxi'],
                   'financial' : ['atm', 'bank', 'bureau_de_change'],
                   'healthcare' : ['clinic', 'dentist', 'doctors', 'hospital', 'nursing_home', 'pharmacy', 'veterinary'],
                   'entertainment' : ['arts_centre', 'casino', 'cinema', 'community_centre', 'events_venue', 'fountain', 'nightclub', 'planetarium', 'public_bookcase', 'studio', 'theatre'],
                   'public service' : ['courthouse', 'fire_station', 'police', 'post_box', 'post_office', 'townhall', 'prison'],
                   'facilities' : ['bbq', 'bench', 'drinking_water', 'shelter', 'shower', 'telephone', 'toilets'],
                   'waste managment' : ['recycling', 'waste_basket', 'waste_disposal'],
                   'misc_amenity' : ['childcare', 'crematorium', 'funeral_hall', 'grave_yard', 'internet_cafe', 'marketplace', 'place_of_worship' ]
                   },
      'buildings' : { 'accomodation': ['apartments', 'barracks', 'bungalow', 'farm', 'hotel', 'house', 'residential', 'detached', 'semidetached_house'],
                      'commercial' :['commercial', 'industrial','office', 'retail', 'supermarket'],
                      'agricultural' : ['barn', 'greenhouse', 'stable', 'sty'],
                      'sports' : ['pavilion', 'sports_hall', 'stadium'],
                     'cars' : ['garage'],
                     'misc_building' : ['military']

                     },
      'shop' : {
          'food/beverage' : ['alcohol', 'bakery','beverages', 'butcher', 'cheese', 'chocolate', 'coffee', 'confectionary', 'convenience', 'deli', 'dairy', 'farm', 'greengrocer', 'pastry', 'pasta', 'seafood', 'spices', 'tea', 'wine', 'water'],
          'general': ['department_store', 'general', 'kiosk', 'mall', 'supermarket', 'wholesale'],
          'clothing': ['baby_goods', 'bags', 'boutique', 'clothes', 'fabric', 'jewelry', 'leather', 'sewing', 'shoes', 'tailor', 'watches', 'wool'],
          'charity' : ['charity'],
          'heath/beauty' : ['beauty', 'chemist', 'cosmetics', 'hairdresser', 'massage', 'hearing_aids', 'herbalist', 'perfumery', 'tattoo'],
          'household' : ['agrarian', 'appliance', 'doityourself', 'electrical', 'energy', 'fireplace', 'florist', 'garden_centre', 'gas', 'hardware', 'houseware', 'locksmith', 'paint', 'security', 'trade'],
          'furniture' : ['antiques', 'bed', 'candles', 'carpet', 'curtain', 'doors', 'flooring', 'furniture', 'interior_decoration', 'kitchen', 'lighting', 'tiles'],
          'electronics' : ['computer', 'electronics', 'hifi', 'mobile_phone', 'radiotechnics'],
          'outdoors' : ['atv', 'bicycle','boat', 'car', 'car_repair', 'car_parts', 'caravan', 'fuel', 'fishing', 'golf', 'hunting', 'jetski', 'motorcycle', 'outdoor', 'scuba_diving', 'ski', 'sports', 'tyres'],
          'hobbies' : ['art', 'camera', 'collector', 'craft', 'frame', 'games', 'model', 'music', 'musical_instrument', 'photo', 'trophy', 'video', 'video_games'],
          'stationery/books' : ['anime', 'books', 'gift', 'lottery', 'newsagent', 'stationery', 'ticket'],
          'misc_shops' : ['bookmaker', 'cannabis', 'copyshop', 'dry_cleaning', 'e-cigarette', 'funeral_directors', 'insurance', 'laundry', 'party', 'pawnbroker', 'pest_control', 'pet', 'pet_grooming', 'pyrotechnics', 'religion', 'storage_rental', 'tobacco', 'toys', 'travel_agency', 'weapons']
      } ,

      'tourism' : ['alpine_hut', 'apartment', 'aquarium', 'artwork', 'attraction', 'camp_site', 'caravan_site', 'chalet', 'gallery', 'guest_house', 'hostel', 'hotel', 'motel', 'museum', 'theme_park', 'wilderness_hut', 'zoo'],

      'aerialway':['cable_car', 'gondola','mixed_lift', 'chair_lift', 'drag_lift', 'goods', 'station'],
      'aeroway':['aerodrome', 'aircraft_crossing', 'heliport', 'runway', 'terminal'],
      'barrier' : ['city_wall', 'ditch', 'fence', 'guardrail', 'hedge', 'wall'],
      'boundary' : ['	administrative', 'border_zone', '	forest', 'national_park', 'protected_area'],
      'craft' : [],
      'emergency' : [],
      'geological' : ['moraine', 'outcrop', 'volcanic_lava_field'],
      'highway' :  ['motorway', 'trunks', 'primary', 'secondary'],
      'historic' : ['archaeological_site', 'building', 'castle', 'church', 'fort', 'manor', 'memorial', 'monument'],
      'landuse' : ['commercial', 'industrial', 'construction', 'residential', 'retail', 'farmland', 'forest', 'meadow', 'orchard', 'vineyard', 'basin', 'cemetery', 'landfill', 'military', 'quarry', 'railway' ],
      'leisure' : ['amusement_arcade', 'beach_resort', 'fishing', 'fitness_centre', 'garden', 'nature_reserve', 'playground', 'swimming_pool'],
      'natural' : ['tree', 'beach', 'geyser', 'hot_spring', 'spring', 'cliff', 'hill', 'volcano'],
      'public_transport' : ['station'],
      'water' : ['river', 'canal', 'ditch', 'lake', 'pond', 'lagoon', 'wastewater']
  }

  tagdf = pd.DataFrame()

  for key in tags:
      if key in pois:
        tdf = pois[pois[key].notnull()]
        if key in ['amenity', 'buildings', 'shop']:
          for key1 in tags[key]:
            ttdf = tdf[tdf[key].isin(tags[key][key1])]
            latitude = ttdf["geometry"].centroid.y
            longitude = ttdf["geometry"].centroid.x
            tagdf = pd.concat([tagdf, pd.DataFrame(list(zip([key1]* len(ttdf),ttdf[key], latitude, longitude)),
              columns=['category', 'tag','latitude', 'longitude'])])
        elif key in ['craft', 'emergency']:      
          ttdf = tdf
          latitude = ttdf["geometry"].centroid.y
          longitude = ttdf["geometry"].centroid.x
          tagdf= pd.concat([tagdf , pd.DataFrame(list(zip([key]* len(ttdf),ttdf[key], latitude, longitude)),
            columns=['category', 'tag','latitude', 'longitude'])])
        else:
          ttdf =  tdf[tdf[key].isin(tags[key])]
          latitude = ttdf["geometry"].centroid.y
          longitude = ttdf["geometry"].centroid.x
          tagdf= pd.concat([tagdf , pd.DataFrame(list(zip([key]* len(ttdf),ttdf[key], latitude, longitude)),
            columns=['category', 'tag','latitude', 'longitude'])])
  return tagdf

def get_vector_distance(row,df3,distance): #This function will get a training vector of features tags times their inverse distance for a particular row.

      tags = sorted(set(df3['tag']))
      categories = sorted(set(df3['category']))
      p_types = {'F':1, 'T':2, 'S':3, 'D':4, 'O':5}
      n_flag = {'Y':1,'N':2}
      t_type = {'L':1, 'F':2}
      tdf3 = df3[(abs(df3['latitude'] - row[11]) <= distance) & (abs(df3['longitude'] - row[12]) <= distance)]
      vector = []
      vector.append(int(row[1].strftime("%Y%m%d")))
      vector.append(p_types[row[3]])
      vector.append(n_flag[row[4]])
      vector.append(t_type[row[5]])
      vector.append(row[11])
      vector.append(row[12])

      for i,tag in enumerate(tags):
        if tag not in tdf3['tag'].unique():
          vector.append(0)
        else:
          ttdf3 = tdf3[tdf3['tag'] == tag]
          invd = (ttdf3.apply(lambda x: 1/(math.hypot(row[11]-x['longitude'], row[12] - x['latitude']) + 0.000001), axis =1)).sum()
          vector.append(invd)   
      return vector 

def get_vector_count(row,df3,distance): #This function will get a training vector of features tag counts for a particular row.

      tags = sorted(set(df3['tag']))
      categories = sorted(set(df3['category']))
      p_types = {'F':1, 'T':2, 'S':3, 'D':4, 'O':5}
      n_flag = {'Y':1,'N':2}
      t_type = {'L':1, 'F':2}
      tdf3 = df3[(abs(df3['latitude'] - row[11]) <= distance) & (abs(df3['longitude'] - row[12]) <= distance)]
      vector = []
      vector.append(int(row[1].strftime("%Y%m%d")))
      vector.append(p_types[row[3]])
      vector.append(n_flag[row[4]])
      vector.append(t_type[row[5]])
      vector.append(row[11])
      vector.append(row[12])

      for i,tag in enumerate(tags):
        if tag not in tdf3['tag'].unique():
          vector.append(0)
        else:
          ttdf3 = tdf3[tdf3['tag'] == tag]
          invd = (ttdf3.apply(lambda x: 1, axis = 1)).sum()
          vector.append(invd)
      return vector   

def get_vector_inv_cat(row,df3, distance):  #This function will get a training vector of features categories times their inverse distance for a particular row.

      tags = sorted(set(df3['tag']))
      categories = sorted(set(df3['category']))
      p_types = {'F':1, 'T':2, 'S':3, 'D':4, 'O':5}
      n_flag = {'Y':1,'N':2}
      t_type = {'L':1, 'F':2}
      tdf3 = df3[(abs(df3['latitude'] - row[11]) <= distance) & (abs(df3['longitude'] - row[12]) <= distance)]
      vector = []

      vector.append(int(row[1].strftime("%Y%m%d")))
      vector.append(p_types[row[3]])
      vector.append(n_flag[row[4]])
      vector.append(t_type[row[5]])
      vector.append(row[11])
      vector.append(row[12])

      for i,cat in enumerate(categories):
        if cat not in tdf3['category'].unique():
          vector.append(0)
        else:
          ttdf3 = tdf3[tdf3['category'] == cat]
          invd = (ttdf3.apply(lambda x: 1/(math.hypot(row[11]-x['longitude'], row[12] - x['latitude']) + 0.000001), axis =1)).sum()
          vector.append(invd)  
      return vector 

def get_vector_count_cat(row,df3, distance):  #This function will get a training vector of features categories counts for a particular row.

      tags = sorted(set(df3['tag']))
      categories = sorted(set(df3['category']))
      p_types = {'F':1, 'T':2, 'S':3, 'D':4, 'O':5}
      n_flag = {'Y':1,'N':2}
      t_type = {'L':1, 'F':2}
      tdf3 = df3[(abs(df3['latitude'] - row[11]) <= distance) & (abs(df3['longitude'] - row[12]) <= distance)]
      vector = []

      vector.append(int(row[1].strftime("%Y%m%d")))
      vector.append(p_types[row[3]])
      vector.append(n_flag[row[4]])
      vector.append(t_type[row[5]])
      vector.append(row[11])
      vector.append(row[12])

      for i,cat in enumerate(categories):
        if cat not in tdf3['category'].unique():
          vector.append(0)
        else:
          ttdf3 = tdf3[tdf3['category'] == cat]
          invd = (ttdf3.apply(lambda x: 1, axis =1)).sum()
          vector.append(invd)
      return vector 
    
def vec_app( df2, distance, pois, func_name = get_vector_distance): #This applies one of the four functions to the entire dataframe. Distance is a measure of the bounding box for each house.
    #Please run this function to get the training dataset for the entire table
  df3 = get_tags(pois)
  vectors = df2.apply(lambda x : func_name(x, df3, distance), axis=1)
  return vectors    

def similarity_price_sort(vectors): #This function gives us a similarity matrix of features. 
    #I have zoomed in to the first and last entries of the vector, as well as displayed a matrix without those entries.
    #A low value indicates high similarity
    vector = np.array([np.array(xi) for xi in vectors])
    distancedf = (distance_matrix(vector, vector))

    fig = plt.figure(figsize=(15,15))
    ax = fig.subplots(2, 2)

    im = ax[0, 0].imshow(distancedf, cmap='rainbow')
    plt.colorbar(im, ax=ax[0, 0])
    tenth = len(vector)/10
    im = ax[0, 1].imshow(distancedf[0:len(vector)//10, 0:len(vector)//10], cmap='rainbow')
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(distancedf[len(vector)//10:9*len(vector)//10, len(vector)//10:9*len(vector)//10], cmap='rainbow')
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(distancedf[99*len(vector)//100:, 99*len(vector)//100:], cmap='rainbow')
    plt.colorbar(im, ax=ax[1, 1])
    
    fig.text(0.5,0.01, "Euclidean Distance of Features", ha="center", va="center")
    plt.show()
    plt.savefig('prices_similarity', bbox_inches = 'tight')
    
def cross_corr(vectors): # This function returns the cross correlation matrix for the features
    vector = np.array([np.array(xi) for xi in vectors])
    corr = pd.DataFrame(vector).corr()
    fig = plt.figure(figsize=(7,7))
    ax = fig.subplots(1, 1)
    im = ax.imshow(corr, cmap='rainbow')
    plt.colorbar(im, ax=ax)
    fig.text(0.5,0.01, "Correlation Matrix of Features", ha="center", va="center")  
    
def correlation_with_price(vectors, df2): #This returns a graph that plots the correlation values of the features with the price
    vector = np.array([np.array(xi) for xi in vectors])
    vec_tran = vector.transpose()
    vec_price = np.array(df2['price'])
    corr_price = [np.corrcoef(a, vec_price)[0,1] for a in vec_tran]
    plt.scatter( [i for i in range(len(corr_price))], corr_price) 
    plt.title( "Correlation with Price of Features")    
    
def plot_price_similarity(df2): #This function returns the similarity matrix of the price data. Certain parts have been zoomed in on like before.
    b = df2['price']
    a = np.array(b)
    a = a.reshape(len(a), 1)

    distancedf = (distance_matrix(a, a))
    fig = plt.figure(figsize=(15,15))
    ax = fig.subplots(2, 2)

    im = ax[0, 0].imshow(distancedf, cmap='rainbow')
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(distancedf[0:len(b)//10, 0:len(b)//10], cmap='rainbow')
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(distancedf[len(b)//10:9*len(b)//10, len(b)//10:9*len(b)//10], cmap='rainbow')
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(distancedf[9*len(b)//10:, 9*len(b)//10:], cmap='rainbow')
    plt.colorbar(im, ax=ax[1, 1])
    fig.text(0.5,0.01, "Euclidean Distance of Prices", ha="center", va="center")
    plt.show()

    plt.savefig('prices_similarity_cam_2017', bbox_inches = 'tight')
    
    
def spec_pois_temp(df2, pois, key, distance, tag=None): #This function returns the number of a specific point of interest within a certain distance

    def spec_pois(pois, key, tag=None):
        tagdf = pd.DataFrame()
        
        if key in pois:
          tdf = pois[pois[key].notnull()]
          if tag:
              ttdf = tdf[tdf[key] == tag]
              tagdf = pd.DataFrame(list(zip(key* len(ttdf),tag, ttdf["geometry"].centroid.y, ttdf["geometry"].centroid.x)),
                columns=['category', 'tag','latitude', 'longitude'])
          else:      
            ttdf = tdf
            tagdf= pd.DataFrame(list(zip([key]* len(ttdf),ttdf[key], ttdf["geometry"].centroid.y, ttdf["geometry"].centroid.x)),
              columns=['category', 'tag','latitude', 'longitude'])
        
        return tagdf

    def spec_pois_sum(row, tagdf ,distance):
        return len(tagdf[(abs(tagdf['latitude'] - row['lattitude']) <= distance) & (abs(tagdf['longitude'] - row['longitude']) <= distance)]) 

    tagdf = spec_pois(pois, key, tag)
    sums = df2.apply(lambda x : spec_pois_sum(x, tagdf, 0.02) , axis = 1)
    return sums    

def spec_pois_num(df2, pois, key, distance, tag=None): # This function plots a similarity matrix for the result of the previous function
    j = np.array(spec_pois_temp(df2, pois, key, distance, tag))
    j = j.reshape(len(j), 1)
    distancedf = (distance_matrix(j, j))
    fig = plt.figure(figsize=(7,7))
    ax = fig.subplots(1, 1)
    im = ax.imshow(distancedf, cmap='rainbow')
    plt.colorbar(im, ax=ax)


def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
