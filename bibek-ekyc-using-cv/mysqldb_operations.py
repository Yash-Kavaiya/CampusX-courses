import mysql.connector
import streamlit as st
import pandas as pd

# CREATE TABLE users(  
#     id VARCHAR(255) NOT NULL PRIMARY KEY,
#     create_time DATETIME COMMENT 'Create Time',
#     name VARCHAR(255),
#     father_name VARCHAR(255),
#     dob DATETIME,
#     id_type VARCHAR(255) NOT NULL,
#     embedding BLOB
# )


# Establish a connection to MySQL Server

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1105",
    database="ekyc"


)
mycursor=mydb.cursor()
print("Connection Established")


def insert_records(text_info):
    sql = "INSERT INTO users(id, name, father_name, dob, id_type, embedding) VALUES (%s, %s, %s, %s, %s, %s)"
    value = (text_info['ID'],
        text_info['Name'],
        text_info["Father's Name"],
        text_info['DOB'],  # Make sure this is formatted as a string 'YYYY-MM-DD'
        text_info['ID Type'],
        str(text_info['Embedding']))
    mycursor.execute(sql, value)
    mydb.commit()

def fetch_records(text_info):
    sql = "SELECT * FROM users WHERE id =%s"
    value = (text_info['ID'],)
    mycursor.execute(sql, value)
    result = mycursor.fetchall()
    if result:
        df = pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
        return df
    else:
        return pd.DataFrame() 

def check_duplicacy(text_info):
    is_duplicate = False
    df =  fetch_records(text_info)
    if df.shape[0]>0:
        is_duplicate = True
    return is_duplicate
    
    