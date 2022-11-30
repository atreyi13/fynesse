from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

import yaml
import pymysql
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """
def hello_world():
    print("This is my first pip package!")
    
def store_credentials():
    @interact_manual(username=Text(description="Username:"), 
                    password=Password(description="Password:"))
    def write_credentials(username, password):
        with open("credentials.yaml", "w") as file:
            credentials_dict = {'username': username, 
                                'password': password}
            yaml.dump(credentials_dict, file)
            
def create_conn():
    database_details = {"url": 'database-ac2354.cgrre17yxw11.eu-west-2.rds.amazonaws.com', 
                    "port": 3306}
    with open("credentials.yaml") as file:
      credentials = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["url"]

    def create_connection(user, password, host, database, port=3306):
        conn = None
        try:
            conn = pymysql.connect(user=user,
                                  passwd=password,
                                  host=host,
                                  port=port,
                                  local_infile=1,
                                  db=database
                                  )
        except Exception as e:
            print(f"Error connecting to the MariaDB Server: {e}")
        return conn
    conn = create_connection(user=credentials["username"], 
                         password=credentials["password"], 
                         host=database_details["url"],
                         database="uk_house_prices")
    return conn              
    
def create_pp_data(conn): #This will empty pp_data.Please don't run this
    c = conn.cursor()

    c.execute('''
                DROP TABLE IF EXISTS `pp_data`;'''
              )

    c.execute('''
  CREATE TABLE IF NOT EXISTS `pp_data` (
  `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,
  `price` int(10) unsigned NOT NULL,
  `date_of_transfer` date NOT NULL,
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
  `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,
  `street` tinytext COLLATE utf8_bin NOT NULL,
  `locality` tinytext COLLATE utf8_bin NOT NULL,
  `town_city` tinytext COLLATE utf8_bin NOT NULL,
  `district` tinytext COLLATE utf8_bin NOT NULL,
  `county` tinytext COLLATE utf8_bin NOT NULL,
  `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,
  `record_status` varchar(2) COLLATE utf8_bin NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;''')
              
    c.execute('''
    CREATE INDEX `pp.postcode` USING HASH
  ON `pp_data`
    (postcode);
CREATE INDEX `pp.date` USING HASH
  ON `pp_data` 
    (date_of_transfer);''')              

    conn.commit()
    return   

def load_pp_data(filenames,conn): #This will put data in pp_data. Please don't run this
    c = conn.cursor()
    for filename in filenames:
      c.execute('''
          LOAD DATA LOCAL INFILE %s INTO TABLE `pp_data`
          FIELDS TERMINATED BY ',' 
          OPTIONALLY ENCLOSED BY '"'
          LINES STARTING BY '' TERMINATED BY '\n';
                '''
                (filename))

    c.execute('''
    ALTER TABLE `pp_data`
    DROP COLUMN `db_id`;''')
              
    c.execute('''
    ALTER TABLE `pp_data` 
    ADD `db_id` bigint(20) NOT NULL AUTO_INCREMENT primary key;''')              

    conn.commit()
    return


def create_postcode_data(conn): #This will empty postcode_data. Please don't run this
    c = conn.cursor()

    c.execute('''
               DROP TABLE IF EXISTS `postcode_data`;'''
              )

    c.execute('''
  CREATE TABLE IF NOT EXISTS `postcode_data` (
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `status` enum('live','terminated') NOT NULL,
  `usertype` enum('small', 'large') NOT NULL,
  `easting` int unsigned,
  `northing` int unsigned,
  `positional_quality_indicator` int NOT NULL,
  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
  `lattitude` decimal(11,8) NOT NULL,
  `longitude` decimal(10,8) NOT NULL,
  `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,
  `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,
  `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,
  `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,
  `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,
  `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,
  `outcode` varchar(4) COLLATE utf8_bin NOT NULL,
  `incode` varchar(3)  COLLATE utf8_bin NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;''')
              
    c.execute('''
    CREATE INDEX `po.postcode` USING HASH
  ON `postcode_data`
    (postcode);''')              

    conn.commit()
    return

def load_postcode_data(conn): #This will load data in postcode_data. Please don't run this
    c = conn.cursor()

    c.execute('''
               LOAD DATA LOCAL INFILE 'open_postcode_geo.csv' INTO TABLE `postcode_data`
               FIELDS TERMINATED BY ',' 
               OPTIONALLY ENCLOSED BY '"'
               LINES STARTING BY '' TERMINATED BY '\n';'''
              )

    c.execute('''
    ALTER TABLE `postcode_data`
    DROP COLUMN `db_id`;''')
              
    c.execute('''
    ALTER TABLE `postcode_data` 
    ADD `db_id` bigint(20) NOT NULL AUTO_INCREMENT primary key;''')              

    conn.commit()
    return

def create_price_coord_data(conn): #This will empty prices_coordinates_data. Please run this before querying for new data.
    c = conn.cursor()

    c.execute('''
              DROP TABLE IF EXISTS `prices_coordinates_data`;'''
              )

    c.execute('''
    CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
      `price` int(10) unsigned NOT NULL,
      `date_of_transfer` date NOT NULL,
      `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
      `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
      `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
      `locality` tinytext COLLATE utf8_bin NOT NULL,
      `town_city` tinytext COLLATE utf8_bin NOT NULL,
      `district` tinytext COLLATE utf8_bin NOT NULL,
      `county` tinytext COLLATE utf8_bin NOT NULL,
      `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
      `lattitude` decimal(11,8) NOT NULL,
      `longitude` decimal(10,8) NOT NULL,
      `db_id` bigint(20) unsigned NOT NULL
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;''')
              
    c.execute('''
    ALTER TABLE `prices_coordinates_data`
    ADD PRIMARY KEY (`db_id`);''')


              
    c.execute('''
    ALTER TABLE `prices_coordinates_data`
    MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;
              ''')                 

    conn.commit()
    return
    
def access_city(county,city,datestart,dateend,country, conn):#This gets prices-coordinates data for a particular city from start date to an end date
    c = conn.cursor()
    c.execute('''
    INSERT INTO `prices_coordinates_data`(`price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`, `country`, `lattitude`, `longitude`)
    (SELECT pp.`price`, pp.`date_of_transfer`, pp.`postcode`, pp.`property_type`, pp.`new_build_flag`, pp.`tenure_type`,
                          pp.`locality`, pp.`town_city`, pp.`district`, pp.`county`, pc.`country`, pc.`lattitude`, pc.`longitude` FROM
                        (SELECT `price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`   FROM `pp_data` 
                          WHERE `county` = %s and `town_city` = %s and `date_of_transfer` >= DATE %s and  `date_of_transfer` <= DATE %s) pp
                    INNER JOIN 
                        (SELECT `postcode`, `country`, `lattitude`, `longitude` FROM `postcode_data` WHERE `country` = %s) pc
                    ON
                        pp.`postcode` = pc.`postcode`);''', (county, city, datestart, dateend, country))
                            
    conn.commit()
    df = pd.read_sql("SELECT * FROM `prices_coordinates_data`", conn).set_index('db_id') 
    return df    

def access_county(county,datestart,dateend,country, conn): #This gets prices-coordinates data for a particular county from start date to an end date
    c = conn.cursor()
    c.execute('''
    INSERT INTO `prices_coordinates_data`(`price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`, `country`, `lattitude`, `longitude`)
    (SELECT pp.`price`, pp.`date_of_transfer`, pp.`postcode`, pp.`property_type`, pp.`new_build_flag`, pp.`tenure_type`,
                          pp.`locality`, pp.`town_city`, pp.`district`, pp.`county`, pc.`country`, pc.`lattitude`, pc.`longitude` FROM
                        (SELECT `price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`   FROM `pp_data` 
                          WHERE `county` = %s and `date_of_transfer` >= DATE %s and  `date_of_transfer` <= DATE %s) pp
                    INNER JOIN 
                        (SELECT `postcode`, `country`, `lattitude`, `longitude` FROM `postcode_data` WHERE `country` = %s) pc
                    ON
                        pp.`postcode` = pc.`postcode`);''', (county, datestart, dateend, country))
                            
    conn.commit()
    df = pd.read_sql("SELECT * FROM `prices_coordinates_data`", conn).set_index('db_id') 
    return df

def access_for_prediction(latstart, latend, longstart, longend, datestart,dateend, property_type, conn): #This gets prices-coordinates data for a particular county from start date to an end date
    c = conn.cursor()
    c.execute('''
    INSERT INTO `prices_coordinates_data`(`price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`, `country`, `lattitude`, `longitude`)
    (SELECT pp.`price`, pp.`date_of_transfer`, pp.`postcode`, pp.`property_type`, pp.`new_build_flag`, pp.`tenure_type`,
                          pp.`locality`, pp.`town_city`, pp.`district`, pp.`county`, pc.`country`, pc.`lattitude`, pc.`longitude` FROM
                        (SELECT `price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`,
                          `locality`, `town_city`, `district`, `county`   FROM `pp_data` 
                          WHERE `property_type` = %s and `date_of_transfer` >= DATE %s and  `date_of_transfer` <= DATE %s) pp
                    INNER JOIN 
                        (SELECT `postcode`, `country`, `lattitude`, `longitude` FROM `postcode_data` 
                        WHERE `lattitude` >=  %s and  `lattitude` <=  %s  and  `longitude` >=  %s  and  `longitude` <=  %s ) pc
                    ON
                        pp.`postcode` = pc.`postcode`);''', (property_type, datestart, dateend, latstart, latend, longstart, longend))
    conn.commit()
    df = pd.read_sql("SELECT * FROM `prices_coordinates_data`", conn).set_index('db_id') 
    return df    
    
def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

