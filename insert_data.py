import sys
import pymysql
import time 
import numpy as np
def insert_data(Nam, A1, A2, A3, A4, A5):
	dat=time.strftime('%Y-%m-%d')
	max_val=[A1,A2,A3,A4,A5]
	count=np.count_nonzero(max_val)
	if count>2:
		count=1
	else:
		count=0
	print(count)
	db = pymysql.connect("localhost","root","","attendance_system" )
	# prepare a cursor object using cursor() method
	cursor = db.cursor()
	sql = """INSERT INTO `bsc_csit_7_a`(Date, Name, Auth1, Auth2, Auth3, Auth4, Auth5, Remarks) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""#'2054-2-2','akash',1,1,1,1,1,1
	# execute SQL query using execute() method.
	cursor.execute(sql,(dat,Nam,A1,A2,A3, A4, A5, count))
	db.commit()
	db.close()
# insert_data('dagina',0,1,1,0,0)